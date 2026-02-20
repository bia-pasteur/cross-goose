import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import find_objects
from torch import nn

from crossgoose.cellpose.dynamics import _extend_centers_gpu
from crossgoose.mask_utils import CenterMethod, get_centers


def sample_points_torch_batch(
    raster: torch.Tensor,
    pts: torch.Tensor,
    interpolation: str = 'bilinear'
):
    """_summary_

    Args:
        T (torch.Tensor): tensor of dim (B,C,H,W)
        pts (torch.Tensor): points of shape (N,3), where pts[:,0] is batch index
    """

    assert len(raster.shape) == 4

    batch_size, c, h, w = raster.shape

    pts_per_batch = [int(torch.sum(pts[:, 0] == i)) for i in range(batch_size)]
    max_pts = max(pts_per_batch)

    size_t = torch.tensor([w, h], device=pts.device)-1
    grid = torch.zeros((batch_size, 1, max_pts, 2),
                       dtype=torch.float, device=pts.device)
    grid_indices = []
    # grid_mask = torch.zeros((batch_size,1,max_pts),dtype=torch.bool)
    for b in range(batch_size):
        pts_b = pts[pts[:, 0] == b]
        n_pts = pts_b.shape[0]
        grid[b, 0, :n_pts] = pts_b[:, [2, 1]]
        grid_indices.append(np.arange(n_pts))

    grid = (grid / size_t[None]) * 2 - 1

    gs = nn.functional.grid_sample(
        input=raster,
        grid=grid,
        mode=interpolation,
        padding_mode='zeros',
        align_corners=True
    )
    # value of shape (b,c,1,max_pts) - > (N,c)
    values = torch.concat([
        gs[b, :, 0, grid_indices[b]].permute(1, 0) for b in range(batch_size)
    ], dim=0)

    assert values.shape[0] == pts.shape[0]

    return values


def extented_diffusion(
    masks_onehot,
    device=None,
    niter=None,
    alpha_out: float = 0.95,
    return_meds: bool = False,
    center_method: CenterMethod = 'dist'
):
    timings = {}
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
            'mps') if torch.backends.mps.is_available() else None

    if masks_onehot is not None:
        assert len(masks_onehot.shape) == 3
        n_lbl_tot = masks_onehot.shape[0]
        masks_oh = masks_onehot
        non_zero_masks = np.max(masks_oh, axis=(1, 2)) > 0
    else:
        raise NotImplementedError("should provide masks_onehot")

    if masks_oh.max() > 0:

        try:
            if non_zero_masks is None:
                mask_oh_nz = masks_oh
            else:
                mask_oh_nz = masks_oh[non_zero_masks]
            # start = time.perf_counter()
            n_label, Ly0, Lx0 = mask_oh_nz.shape
            masks_padded = torch.from_numpy(
                mask_oh_nz.astype("int64")).to(device)
            masks_padded = F.pad(masks_padded, (1, 1, 1, 1))
            # timings['pad_masks'] = time.perf_counter() - start

            # start = time.perf_counter()
            # get center-of-mass within cell
            slices = [find_objects(mask_oh_nz[i])[0] for i in range(n_label)]
            assert len(slices) > 0
            # timings['find_objects'] = time.perf_counter() - start

            # start = time.perf_counter()
            centers, _ = get_centers(
                mask_oh_nz, slices, method=center_method, one_hot_masks=True)
            meds_p = torch.from_numpy(centers).to(device).long()
            meds_p += 1  # for padding
            # timings['get_centers'] = time.perf_counter() - start

            # run diffusion
            n_iter = int(
                2*max(masks_padded.shape[1:])/alpha_out) if niter is None else niter

            # start = time.perf_counter()
            T = torch.zeros(masks_padded.shape,
                            dtype=torch.double, device=device)

            T = T.unsqueeze(1)
            masks_padded_t = masks_padded.unsqueeze(1)
            diff_fac = masks_padded_t + (1-masks_padded_t) * alpha_out

            hsm = torch.jit.script(HeatStepModule(
                diff_fac=diff_fac,
                meds_idx_0=torch.arange(n_label),
                meds_idx_1=meds_p[:, 0],
                meds_idx_2=meds_p[:, 1],
                device=device
            ))
            # timings['prep_T'] = time.perf_counter() - start

            # start = time.perf_counter()
            for i in range(n_iter):
                T = hsm(T)
            # timings['simulation_loop'] = time.perf_counter() - start

            if not return_meds:
                del meds_p
            else:
                meds_p = meds_p.cpu().numpy()

            # start = time.perf_counter()
            T = T[:, 0].double()
            # gradient positions
            dx = T[:, 1:-1, 2:] - T[:, 1:-1, :-2]
            dy = T[:, 2:, 1:-1] - T[:, :-2, 1:-1]
            mu_torch = np.stack((dy.cpu(), dx.cpu()), axis=1)

            # mu = mu_torch.astype("float64")
            mu = mu_torch

            # norm = (1e-8 + (mu**2).sum(axis=1, keepdims=True)**0.5)
            norm = np.linalg.norm(mu, axis=1, ord=2, keepdims=True)
            norm = np.tile(norm, reps=(1, 2, 1, 1))
            mask = norm != 0.0
            mu[mask] = mu[mask] / norm[mask]
            mu[~mask] = 0

            mu = np.clip(mu, -1, 1)

            T = T.cpu().numpy()[:, 1:-1, 1:-1]
        except torch.OutOfMemoryError as e:
            raise e

    else:
        _, Ly0, Lx0 = masks_oh.shape
        # no masks, return empty flows
        mu = np.zeros((masks_oh.shape[0], 2, Ly0, Lx0))
        T = np.zeros((masks_oh.shape[0], Ly0, Lx0))
        meds_p = None

    if non_zero_masks is not None:
        mu_full = np.zeros_like(mu, shape=(n_lbl_tot,) + mu.shape[1:])
        mu_full[non_zero_masks] = mu

        T_full = np.zeros_like(T, shape=(n_lbl_tot,) + T.shape[1:])
        T_full[non_zero_masks] = T

        T = T_full
        mu = mu_full

        if return_meds:
            raise NotImplementedError(
                'does not support return_meds and non_zero_masks')

    if return_meds:
        return mu, T, meds_p

    # pprint(timings)

    return mu, T, None


class HeatStepModule(nn.Module):
    def __init__(self, diff_fac, meds_idx_0, meds_idx_1, meds_idx_2, device):
        super().__init__()
        self.khole = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]],
                                  dtype=torch.double, device=device).reshape((1, 1, 3, 3))
        self.meds_idx_0 = meds_idx_0.to(device)
        self.meds_idx_1 = meds_idx_1.to(device)
        self.meds_idx_2 = meds_idx_2.to(device)
        self.diff_fac = diff_fac.to(device)

    def forward(self, T: torch.Tensor):
        T[self.meds_idx_0, :, self.meds_idx_1, self.meds_idx_2] += 1.0
        T_conv = torch.nn.functional.conv2d(  # pylint: disable=E1102
            T*self.diff_fac, self.khole, padding=1)
        return (T + T_conv) / 9.0


def large_steps_diffusion(
    step_size: int,
    masks_onehot=None,
    device=None,
    niter=None,
    alpha_out: float = 0.95,
    center_method: CenterMethod = 'graph_center_weighted'
):
    mu, _, _ = extented_diffusion(
        masks_onehot=masks_onehot,
        device=device,
        niter=niter,
        alpha_out=alpha_out,
        return_meds=True,
        center_method=center_method
    )
    mu = torch.from_numpy(mu)

    n_labels, _, h, w = mu.shape
    u = torch.nonzero(torch.ones((n_labels, h, w)))
    u_instance = u[:, 0].long()
    u_coord = u[:, 1:].double()
    del u
    u0 = u_coord.clone()
    min_v = torch.tensor([0, 0])
    max_v = torch.tensor([h-1, w-1])

    for _ in range(step_size):
        ui = u_coord[:, 0].long()
        uj = u_coord[:, 1].long()
        u_coord = torch.clip(u_coord + mu[u_instance, :, ui, uj],
                             min_v, max_v)

    vec = u_coord - u0
    new_mu = torch.zeros_like(mu)
    new_mu[u_instance, :, u0[:, 0].long(), u0[:, 1].long()] = vec

    return new_mu.numpy(), mu.numpy()


def cp_masks_to_flows_gpu(masks, device=torch.device("cpu"), niter=None, center_method: CenterMethod = 'mass'):
    """Convert masks to flows using diffusion from center pixel.

    Center of masks where diffusion starts is defined by pixel closest to median within the mask.

    Args:
        masks (int, 2D or 3D array): Labelled masks. 0=NO masks; 1,2,...=mask labels.
        device (torch.device, optional): The device to run the computation on. Defaults to torch.device("cpu").
        niter (int, optional): Number of iterations for the diffusion process. Defaults to None.

    Returns:
        np.ndarray: A 4D array representing the flows for each pixel in Z, X, and Y.


    Returns:
        A tuple containing (mu, meds_p). mu is float 3D or 4D array of flows in (Z)XY. 
        meds_p are cell centers.
    """
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
            'mps') if torch.backends.mps.is_available() else None

    Ly0, Lx0 = masks.shape

    masks_padded = torch.from_numpy(masks.astype("int64")).to(device)
    masks_padded = F.pad(masks_padded, (1, 1, 1, 1))
    shape = masks_padded.shape

    # get mask pixel neighbors
    y, x = torch.nonzero(masks_padded, as_tuple=True)
    y = y.int()
    x = x.int()
    neighbors = torch.zeros((2, 9, y.shape[0]), dtype=torch.int, device=device)
    yxi = [[0, -1, 1, 0, 0, -1, -1, 1, 1], [0, 0, 0, -1, 1, -1, 1, -1, 1]]
    for i in range(9):
        neighbors[0, i] = y + yxi[0][i]
        neighbors[1, i] = x + yxi[1][i]
    isneighbor = torch.ones((9, y.shape[0]), dtype=torch.bool, device=device)
    m0 = masks_padded[neighbors[0, 0], neighbors[1, 0]]
    for i in range(1, 9):
        isneighbor[i] = masks_padded[neighbors[0, i], neighbors[1, i]] == m0
    del m0, masks_padded

    # get center-of-mass within cell
    slices = find_objects(masks)
    # turn slices into array
    # slices = np.array([
    #     np.array([i, si[0].start, si[0].stop, si[1].start, si[1].stop])
    #     for i, si in enumerate(slices)
    #     if si is not None
    # ])
    centers, ext = get_centers(masks, slices, method=center_method)
    meds_p = torch.from_numpy(centers).to(device).long()
    meds_p += 1  # for padding

    # run diffusion
    n_iter = 2 * ext.max() if niter is None else niter
    mu = _extend_centers_gpu(neighbors, meds_p, isneighbor, shape, n_iter=n_iter,
                             device=device)
    mu = mu.astype("float64")

    # new normalization
    mu /= (1e-60 + (mu**2).sum(axis=0)**0.5)

    # put into original image
    mu0 = np.zeros((2, Ly0, Lx0))
    mu0[:, y.cpu().numpy() - 1, x.cpu().numpy() - 1] = mu

    return mu0, meds_p.cpu().numpy() - 1
