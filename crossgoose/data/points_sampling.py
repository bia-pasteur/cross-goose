from abc import ABC, abstractmethod


import concurrent
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
        flows = torch.stack([torch.zeros_like(flows),flows],axis=1)

        mask = torch.ones(
            size=points.shape[:2],
            dtype=bool,
            device=points.device
        )
        mask[:,0] = False #for simple u0,ut sampling only the flows at flows[:,1] is valid

        return points, flows, mask


# class FlowTracker(PointsSamlper):
