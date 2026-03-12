import concurrent
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor

from crossgoose.gridflow import GridFlow


class PointsSamlper(ABC):
    @abstractmethod
    def sample(self, image: Tensor, labels: Tensor, grid_flow: GridFlow) -> tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError


class RandomOnCell(PointsSamlper):
    def __init__(
        self,
        n_samples: int,
        sigma: float
    ):
        super().__init__()
        self.n_samples = n_samples
        self.sigma = sigma

    def _sample_one_label(
        self,
        l0: Tensor,
        labels: Tensor,
        k: int,
        grid_flow: GridFlow,
        min_bound: Tensor,
        max_bound: Tensor
    ):
        pt_mask = l0 == k
        label_pts = torch.nonzero(labels == k)

        n = int(torch.sum(pt_mask))
        m = label_pts.shape[0]

        samples_idx = torch.randint(
            0, m,
            size=(self.n_samples, n)
        )
        samples = label_pts[samples_idx].float()

        pert = self.sigma * torch.randn_like(samples)
        samples = torch.clamp(
            samples + pert,
            min=min_bound, max=max_bound
        )

        flow = grid_flow.query(
            pos=samples.numpy(),
            label=int(k)
        )
        flow = torch.from_numpy(flow).float()

        return samples, flow, pt_mask

    def sample(
        self,
        image: Tensor,
        labels: Tensor,
        grid_flow: GridFlow
    ):
        h, w = image.shape

        # get all non zero labels
        u0 = torch.nonzero(labels)
        l0 = labels[u0[:, 0], u0[:, 1]]

        # we then generate for each points n_samples points that fall within the same label with perturbation sigma

        min_bound = torch.tensor([0, 0], device=u0.device)
        max_bound = torch.tensor([h, w], device=u0.device)-1

        unique_labels = torch.unique(l0)

        ut = torch.zeros(
            (self.n_samples,)+u0.shape,
            dtype=torch.float)
        flows = torch.zeros(
            (self.n_samples,)+u0.shape,
            dtype=torch.float)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for k in unique_labels:
                futures.append(
                    executor.submit(
                        self._sample_one_label,
                        l0=l0,
                        labels=labels,
                        k=k,
                        grid_flow=grid_flow,
                        min_bound=min_bound,
                        max_bound=max_bound
                    )
                )

            for f in concurrent.futures.as_completed(futures):
                samples, flow, pt_mask = f.result()
                ut[:, pt_mask, :] = samples
                flows[:, pt_mask, :] = flow

        ut = ut.reshape((-1, 2)).float()
        flows = flows.reshape((-1, 2)).float()
        u0 = torch.tile(u0, (self.n_samples, 1)).float()

        points = torch.stack([u0, ut], axis=1)
        # for consistency we fill  flows[:,0] with nonsense and mask it out
        flows = torch.stack([torch.zeros_like(flows), flows], axis=1)

        weights = torch.ones(
            size=points.shape[:2],
            device=points.device
        )
        # for simple u0,ut sampling only the flows at flows[:,1] is valid
        weights[:, 0] = 0
        weights /= weights.sum()

        return points, flows, weights


class RandomOnCellV2(PointsSamlper):
    def __init__(
        self,
        n_samples: int,
        sigma: float
    ):
        super().__init__()
        self.n_samples = n_samples
        self.sigma = sigma

    def sample(
        self,
        image: Tensor,
        labels: Tensor,
        grid_flow: GridFlow
    ):
        h, w = image.shape

        # get all non zero labels
        pts_init = torch.nonzero(labels)
        # sample n_samples
        samples_idx = torch.randint(
            0, pts_init.shape[0],
            size=(self.n_samples,)
        )
        pts_init = pts_init[samples_idx]
        pts_labels = labels[pts_init[:, 0], pts_init[:, 1]]

        # for each point u0, pick a random point in the same instance
        unique_labels = torch.unique(pts_labels)
        label_points = {
            int(k): torch.nonzero(labels == k) for k in unique_labels
        }
        label_points_nb = {int(k): v.shape[0] for k, v in label_points.items()}
        max_pts = np.max(list(label_points_nb.values()))
        # we pick random integers less than the max number of px in an instance, then mod it to the actual number per instance
        pts_t_samples_idx = torch.randint(
            0, max_pts,
            size=(self.n_samples,)
        )

        pts_t = torch.stack([
            label_points[int(k)][pts_t_samples_idx[i] % label_points_nb[int(k)]] for i, k in enumerate(pts_labels)
        ], axis=0).float()

        min_bound = torch.tensor([0, 0], device=pts_init.device)
        max_bound = torch.tensor([h, w], device=pts_init.device)-1
        pert = self.sigma * torch.randn_like(pts_t)
        pts_t = torch.clamp(
            pts_t + pert,
            min=min_bound, max=max_bound
        )

        flows = grid_flow.query_multiple_labels_threaded(
            pos=pts_t.numpy(),
            labels=pts_labels.numpy()
        )
        flows = torch.from_numpy(flows).float()

        points = torch.stack([pts_init, pts_t], axis=1)

        # for consistency we fill  flows[:,0] with nonsense and mask it out
        flows = torch.stack([torch.zeros_like(flows), flows], axis=1)

        weights = torch.ones(
            size=points.shape[:2],
            device=points.device
        )
        # for simple u0,ut sampling only the flows at flows[:,1] is valid
        weights[:, 0] = 0
        weights /= weights.sum()

        return points, flows, weights


class TrajectorySampler(PointsSamlper):
    def __init__(
        self,
        n_steps: int,
        n_samples: int,
        weight_by_location: bool
    ):
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.weight_by_location = weight_by_location

    def sample(
        self,
        image: Tensor,
        labels: Tensor,
        grid_flow: GridFlow
    ):
        h, w = image.shape

        # get all non zero labels
        pts_init = torch.nonzero(labels)
        # sample n_samples
        samples_idx = torch.randint(
            0, pts_init.shape[0],
            size=(self.n_samples,)
        )
        pts_init = pts_init[samples_idx].numpy()
        pts_labels = labels[pts_init[:, 0], pts_init[:, 1]].numpy()

        # create points (N,T,2)
        points = np.zeros(
            (self.n_samples, self.n_steps, 2)
        )
        points[:, 0] = pts_init

        # create flows (N,T,2)
        flows = np.zeros(
            (self.n_samples, self.n_steps, 2)
        )
        flows[:, 0] = grid_flow.query_multiple_labels_threaded(
            pos=points[:, 0],
            labels=pts_labels
        )

        min_bound = np.array([0, 0])
        max_bound = np.array([h, w])-1

        for t in range(self.n_steps-1):
            points[:, t+1] = np.clip(
                points[:, t] + flows[:, t],
                min=min_bound, max=max_bound
            )
            flows[:, t+1] = grid_flow.query_multiple_labels_threaded(
                pos=points[:, t],
                labels=pts_labels
            )

        if self.weight_by_location:
            # we count how many times each pixel is sampled
            # and weight points by the inverse pixel count
            points_int = points.astype(int)
            _, u_inverse, u_counts = np.unique(
                points_int.reshape((-1, 2)), axis=0,
                return_counts=True, return_inverse=True)
            weights = 1 / \
                u_counts[u_inverse].reshape((self.n_samples, self.n_steps))
            weights /= weights.sum()
        else:
            weights = np.ones((self.n_samples, self.n_steps)) / \
                (self.n_samples * self.n_steps)
            weights /= weights.sum()

        points = torch.from_numpy(points).float()
        flows = torch.from_numpy(flows).float()
        weights = torch.from_numpy(weights).float()
        # weights /= weights.sum()

        return points, flows, weights
