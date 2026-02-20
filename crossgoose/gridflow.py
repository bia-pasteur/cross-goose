

from pathlib import Path
from typing import Dict, Literal, Self, Tuple

import numpy as np
import skimage
from scipy.interpolate import interpn
from scipy.ndimage import find_objects
from scipy.spatial import KDTree

from crossgoose.cellpose.dynamics import masks_to_flows_gpu
from crossgoose.dynamics import cp_masks_to_flows_gpu
from crossgoose.mask_utils import CenterMethod
from crossgoose.utils import normalize_vec


class GridFlow:
    def __init__(
        self,
        points: Dict[str, KDTree],
        flows: Dict[str, np.ndarray],
        n_interpol: int
    ):
        self.points = points
        self.flows = flows
        self.n_interpol = n_interpol

    @classmethod
    def from_one_hot(
        self,
        labels_one_hot: np.ndarray,
        n_interpol: int,
        flow_center_method:CenterMethod,
        inside_sub_sampling: int,
        contour_sub_sampling: int,
        contour_method: Literal['marching_squares',
                                'dilation'] = 'marching_squares',
        flow_compute_device:str='cpu'
    ) -> Self:

        n_labels = labels_one_hot.shape[0]
        points: Dict[str, KDTree] = {}
        flows: Dict[str, np.ndarray] = {}
        for k in range(n_labels):
            mask = labels_one_hot[k]
            mask_size = np.sum(mask)
            assert mask_size > 0
            # compute classical cp flow
            # first get the local mask
            slices = find_objects(mask.astype(int))
            assert len(slices) == 1
            slice_k = slices[0]
            padding = 2
            offest_i, offset_j = slice_k[0].start, slice_k[1].start
            offset = np.array([offest_i, offset_j]) - padding

            mask_local = np.pad(mask[slice_k], pad_width=padding)
            mask_local_dil = skimage.morphology.isotropic_dilation(
                mask_local, radius=1)
            flow_local, _ = cp_masks_to_flows_gpu(
                mask_local_dil.astype(int),
                device=flow_compute_device,
                center_method=flow_center_method
            )

            if contour_method == 'dilation':
                contours_local = np.stack(np.nonzero(
                    (mask_local_dil != mask_local)[::contour_sub_sampling, ::contour_sub_sampling]), axis=1) * contour_sub_sampling

            elif contour_method == 'marching_squares':
                ct = skimage.measure.find_contours(
                    mask_local)
                if len(ct) == 0:
                    contours_local = np.zeros((0, 2))
                else:
                    contours_local = np.concat(ct, axis=0)[
                        ::contour_sub_sampling]

            else:
                raise ValueError

            inside = np.stack(np.nonzero(
                mask_local[::inside_sub_sampling, ::inside_sub_sampling]), axis=1) * inside_sub_sampling

            points_k = np.concat([contours_local, inside], axis=0)
            points_k = np.unique(points_k, axis=0)

            points[k+1] = KDTree(points_k + offset[None, :])

            h, w = mask_local.shape
            flows_k = interpn(
                points=(np.arange(h), np.arange(w)),
                values=flow_local.transpose((1, 2, 0)),
                xi=points_k,
                method='cubic'
            )
            flows[k+1] = flows_k

        return GridFlow(
            points=points,
            flows=flows,
            n_interpol=n_interpol
        )

    def to_file(self, file: Path):

        file = Path(file)
        assert file.suffix == '.npz'

        points = {k: v.data.copy() for k, v in self.points.items()}
        flows = {k: v.copy() for k, v in self.flows.items()}

        data_flat = {f'points/{k}': v for k, v in points.items()}
        data_flat.update({f'flows/{k}': v for k, v in flows.items()})

        np.savez_compressed(file, **data_flat, allow_pickle=False)

    @classmethod
    def from_file(self, file: Path, n_interpol) -> Self:
        data = np.load(file)
        points = {}
        flows = {}
        for k, v in data.items():
            kind, label = k.split('/')
            label = int(label)
            if kind == 'points':
                points[label] = KDTree(v)
            elif kind == 'flows':
                flows[label] = v
            else:
                raise KeyError(
                    f'key {k} does not match points/label or flow/label')

        return GridFlow(
            points=points,
            flows=flows,
            n_interpol=n_interpol
        )

    def query(self, x: np.ndarray, label: int) -> np.ndarray:

        distance, nearest_vertex = self.points[label].query(
            x, k=self.n_interpol)

        vec = self.flows[label][nearest_vertex]

        if self.n_interpol > 1:
            # agglomerate results if interpolating
            weights = 1 / np.clip(distance, 1.0, np.inf)
            weights = weights / np.sum(weights)
            vec = np.sum(vec * weights[..., None], axis=-2)

        vec = normalize_vec(vec, axis=-1)

        return vec

    def query_flow_grid(self, label: int, shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        pts = np.reshape(np.mgrid[:h, :w], (2, -1)).transpose()
        flow = self.query(x=pts, label=label).transpose().reshape((2, h, w))
        return flow
