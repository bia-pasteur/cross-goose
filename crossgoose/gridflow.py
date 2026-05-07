""" A compact way to store flows for overlapping labels
"""

import concurrent
import math
from pathlib import Path
from typing import Dict, List, Literal, Self, Tuple

import numpy as np
import skimage
from scipy.interpolate import interpn
from scipy.ndimage import find_objects
from scipy.spatial import KDTree

from crossgoose.cellpose.dynamics import _extend_centers
from crossgoose.mask_utils import CENTER_METHODS, CenterMethod
from crossgoose.utils import normalize_vec


def _single_mask_to_flow(
    mask: np.ndarray,
    center_method: CenterMethod,
    n_iter: int | None = None,
):
    assert np.max(mask) < 2  # we just do one instance
    ly, lx = mask.shape
    y, x = np.nonzero(mask)
    y = y.astype(np.int32)  # is this necessary ?
    x = x.astype(np.int32)

    c_fun = CENTER_METHODS[center_method]
    ymed, xmed = c_fun(mask > 0)

    n_iter = 2 * np.int32(ly + lx) if n_iter is None else n_iter
    T = np.zeros((ly) * (lx), np.float64)
    T = _extend_centers(T, y, x, ymed, xmed, np.int32(lx), np.int32(n_iter))
    dy = T[(y + 1) * lx + x] - T[(y - 1) * lx + x]
    dx = T[y * lx + x + 1] - T[y * lx + x - 1]
    mu = np.zeros((2, ly, lx), np.float64)
    mu[:, y, x] = np.stack((dy, dx))
    mu /= (1e-60 + (mu**2).sum(axis=0)**0.5)

    return mu


def keep_largest_component(
        mask: np.ndarray
) -> np.ndarray:
    labels = skimage.measure.label(mask)
    unique_labels, counts_labels = np.unique(
        labels[mask > 0], return_counts=True)
    max_label = unique_labels[np.argmax(counts_labels)]
    return labels == max_label


def _mask_to_pts_and_flw(
    mask: np.ndarray,
    flow_center_method: CenterMethod,
    inside_sub_sampling: int,
    contour_sub_sampling: int,
    contour_method: Literal['marching_squares', 'dilation']
):
    mask_size = np.sum(mask)
    assert mask_size > 0
    assert len(mask.shape) == 2, mask.shape
    # compute classical cp flow
    # first get the local mask
    slices = find_objects(mask.astype(int))
    assert len(slices) == 1
    slice_k = slices[0]
    padding = 2
    offest_i, offset_j = slice_k[0].start, slice_k[1].start
    offset = np.array([offest_i, offset_j]) - padding

    mask_local = np.pad(mask[slice_k], pad_width=padding)
    # keep largest component to avoid unconnected components
    mask_local = keep_largest_component(mask_local)
    mask_local_dil = skimage.morphology.isotropic_dilation(
        mask_local, radius=1)

    flow_local = _single_mask_to_flow(
        mask=mask_local_dil,
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

    points = points_k + offset[None, :]

    h, w = mask_local.shape
    flows_k = interpn(
        points=(np.arange(h), np.arange(w)),
        values=flow_local.transpose((1, 2, 0)),
        xi=points_k,
        method='cubic'
    )
    flows = flows_k

    return points, flows


class GridFlow:
    def __init__(
        self,
        points: Dict[str, np.ndarray],
        flows: Dict[str, np.ndarray],
        n_interpol: int
    ):
        self.n_interpol = n_interpol

        for k in points.keys():
            assert k in flows
            assert flows[k].shape[0] == points[k].shape[0]

        assert len(flows) == len(points)
        self.points = {k: KDTree(v) for k, v in points.items()}
        self.flows = flows

    def __str__(self):
        return f"GridFlow with {len(self.points)} labels"

    @classmethod
    def from_one_hot(
        self,
        labels_one_hot: np.ndarray,
        n_interpol: int,
        flow_center_method: CenterMethod,
        inside_sub_sampling: int = 4,
        contour_sub_sampling: int = 2,
        contour_method: Literal['marching_squares',
                                'dilation'] = 'marching_squares',
    ) -> Self:
        assert len(labels_one_hot.shape) == 3
        points: Dict[str, KDTree] = {}
        flows: Dict[str, np.ndarray] = {}

        non_zero_labels = np.nonzero(np.sum(labels_one_hot, axis=(1, 2)))[0]

        for k in non_zero_labels:
            mask = labels_one_hot[k]
            assert len(mask.shape) == 2, k
            pts, flw = _mask_to_pts_and_flw(
                mask=mask,
                flow_center_method=flow_center_method,
                inside_sub_sampling=inside_sub_sampling,
                contour_sub_sampling=contour_sub_sampling,
                contour_method=contour_method
            )

            points[k+1] = pts
            flows[k+1] = flw

        return GridFlow(
            points=points,
            flows=flows,
            n_interpol=n_interpol
        )

    @classmethod
    def from_labels(
        self,
        labels: np.ndarray,
        n_interpol: int,
        flow_center_method: CenterMethod,
        inside_sub_sampling: int = 4,
        contour_sub_sampling: int = 2,
        contour_method: Literal['marching_squares',
                                'dilation'] = 'marching_squares',
    ) -> Self:
        assert len(labels.shape) == 2
        points: Dict[str, KDTree] = {}
        flows: Dict[str, np.ndarray] = {}

        non_zero_labels = np.unique(labels)

        for k in non_zero_labels:
            if k != 0:
                mask = labels == k

                pts, flw = _mask_to_pts_and_flw(
                    mask=mask,
                    flow_center_method=flow_center_method,
                    inside_sub_sampling=inside_sub_sampling,
                    contour_sub_sampling=contour_sub_sampling,
                    contour_method=contour_method
                )

                points[k] = pts
                flows[k] = flw

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
                points[label] = v
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

    def query_multiple_labels(self, pos: np.ndarray, labels: np.ndarray) -> np.ndarray:
        assert len(pos.shape) == 2
        assert pos.shape[-1] == 2
        assert pos.shape[0] == labels.shape[0]

        n_x = pos.shape[0]

        unique_labels = np.unique(labels)
        vec = np.empty((n_x, 2))
        for l in unique_labels:
            mask = labels == l
            vec_l = self.query(
                pos=pos[mask],
                label=l
            )
            vec[mask, :] = vec_l

        return vec

    def query_multiple_labels_threaded(self, pos: np.ndarray, labels: np.ndarray) -> np.ndarray:
        assert len(pos.shape) == 2
        assert pos.shape[-1] == 2
        assert pos.shape[0] == labels.shape[0]

        n_x = pos.shape[0]

        unique_labels = np.unique(labels)
        vec = np.empty((n_x, 2))
        masks = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            to_compute = []
            for l in unique_labels:
                mask = labels == l
                masks.append(mask)
                to_compute.append(
                    executor.submit(
                        self.query,
                        pos=pos[mask],
                        label=l
                    )
                )
            for f in concurrent.futures.as_completed(to_compute):
                f: concurrent.futures.Future
                i = to_compute.index(f)
                mask = masks[i]
                vec[mask, :] = f.result()

        return vec

    def query(self, pos: np.ndarray, label: int) -> np.ndarray:
        assert pos.shape[-1] == 2
        if label not in self.points.keys():
            raise KeyError(
                f"label {label} not in Gridflow with labels {self.points.keys()}")

        n_pts_max = self.points[label].n
        ninterpol = min(self.n_interpol, self.points[label].n)
        distance, nearest_vertex = self.points[label].query(
            pos, k=ninterpol)
        # warning missing neighbors are insicated with n_pts_max
        valid_pts = nearest_vertex < n_pts_max
        assert np.all(
            valid_pts), f"{pos.shape=}, {self.points[label].data.shape=}, , {self.n_interpol=}"

        try:
            vec = self.flows[label][nearest_vertex]
        except IndexError as e:
            print(
                f"failed to fetch {nearest_vertex=}, in flows {self.flows[label].shape=}")
            raise e

        if ninterpol > 1:
            # agglomerate results if interpolating
            weights = 1 / np.clip(distance, 1.0, np.inf)
            weights = weights / np.sum(weights)
            vec = np.sum(vec * weights[..., None], axis=-2)

        vec = normalize_vec(vec, axis=-1)

        return vec

    def query_flow_grid(self, label: int, shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        pts = np.reshape(np.mgrid[:h, :w], (2, -1)).transpose()
        flow = self.query(pos=pts, label=label).transpose().reshape((2, h, w))
        return flow

    def get_label_keys(self) -> List[int]:
        return list(self.points.keys())

    def affine_transform(
        self,
        center: Tuple[float, float],
        scale: float = 1.0,
        angle: float = 0.0,
        translate: Tuple[float, float] = (0, 0),
        shear: Tuple[float, float] = (0, 0),
    ) -> Self:

        if shear != (0, 0):
            raise NotImplementedError

        transform_matrix = get_affine_transform_matrix(
            translation=translate,
            center=np.array(center),
            rot=-angle,
            scale=scale
        )

        new_points = {k: _transform_points(v.data, transform_matrix)
                      for k, v in self.points.items()}
        new_flows = {k: _transform_flows(v, angle)
                     for k, v in self.flows.items()}

        return GridFlow(
            points=new_points,
            flows=new_flows,
            n_interpol=self.n_interpol
        )

    def pad(self, padding: Tuple[int, int, int, int]) -> Self:
        offset = np.array([[padding[2], padding[0]]])
        return GridFlow(
            points={k: offset + v.data for k, v in self.points.items()},
            flows=self.flows,
            n_interpol=self.n_interpol
        )

    def crop(self, top: int, left: int, height: int, width: int, drop_oob: bool = False) -> Self:
        offset = np.array([[-top, -left]])
        new_points = {}
        new_flows = {}
        shape_arr = np.array([height, width])
        zeros_arr = np.array([height, width])
        for k in self.points.keys():
            pts = offset + self.points[k].data
            keep = True
            if drop_oob:
                max_bound = np.max(pts, axis=0)
                min_bound = np.min(pts, axis=0)
                if np.all(max_bound > shape_arr) or np.all(min_bound < zeros_arr):
                    keep = False  # the shape is oob
            if keep:
                new_points[k] = pts
                new_flows[k] = self.flows[k]

        return GridFlow(
            points=new_points,
            flows=new_flows,
            n_interpol=self.n_interpol
        )

    def flip(self, mode: Literal['h', 'v'], shape: Tuple[int]) -> Self:
        h, w = shape
        new_flows = {}
        new_points = {}
        for k in self.points.keys():
            new_flows[k] = self.flows[k].copy()
            new_points[k] = self.points[k].data.copy()
            if mode == 'h':
                new_flows[k][:, 1] = -new_flows[k][:, 1]
                new_points[k][:, 1] = w - new_points[k][:, 1]
            elif mode == 'v':
                new_points[k][:, 0] = h - new_points[k][:, 0]
                new_flows[k][:, 0] = -new_flows[k][:, 0]
            else:
                raise ValueError(mode)

        return GridFlow(
            points=new_points,
            flows=new_flows,
            n_interpol=self.n_interpol
        )

    def relabel(self, mapping: Dict[int, int]):
        self.points = {new_key: self.points[old_key]
                       for new_key, old_key in mapping.items()}
        self.flows = {new_key: self.flows[old_key]
                      for new_key, old_key in mapping.items()}

# Affine matrix is : M = T * C * RotateScaleShear * C^-1
# where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
#       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
#       RotateScaleShear is rotation with scale and shear matrix
#
#       RotateScaleShear(a, s, (sx, sy)) =
#       = R(a) * S(s) * SHy(sy) * SHx(sx)
#       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(sx)/cos(sy) - sin(a)), 0 ]
#         [ s*sin(a - sy)/cos(sy), s*(-sin(a - sy)*tan(sx)/cos(sy) + cos(a)), 0 ]
#         [ 0                    , 0
# where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
# SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
#          [0, 1      ]              [-tan(s), 1]


def get_affine_transform_matrix(translation, center, rot, scale):
    # https://docs.pytorch.org/vision/main/_modules/torchvision/transforms/functional.html#affine
    tx, ty = translation
    cx, cy = center
    rot = math.radians(rot)
    a = math.cos(rot)
    b = - math.sin(rot)
    c = math.sin(rot)
    d = math.cos(rot)
    matrix = [a, b, 0.0, c, d, 0.0]
    matrix = [x * scale for x in matrix]
    matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
    matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
    matrix[2] += cx + tx
    matrix[5] += cy + ty
    return np.array(matrix).reshape(2, 3)


def get_rotation_matrix(angle: float, degree: bool = False) -> np.ndarray:
    if degree:
        angle = math.radians(angle)
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


def _transform_points(x: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    x = np.matmul(transform_matrix, np.concat(
        [x, np.ones((x.shape[0], 1))], axis=1).T).T
    return x[:, :2]


def _transform_flows(x: np.ndarray, angle: float) -> np.ndarray:
    # translate and scale have no effect
    if angle != 0.0:
        rot_matrix = get_rotation_matrix(-angle, degree=True)
        x = np.matmul(rot_matrix, x.transpose()).transpose()

    return x


class BatchGridFlow:
    def __init__(
        self,
        gridflows: List[GridFlow]
    ):
        self.gridflows: List[GridFlow] = gridflows

    def __getitem__(self, key) -> GridFlow:
        return self.gridflows[key]

    def batch_query_multiple_labels(
        self,
        pos: np.ndarray,
        batch_indices: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        assert pos.shape[0] == batch_indices.shape[0]
        assert pos.shape[0] == labels.shape[0]
        n_pts = pos.shape[0]
        unique_batch_indices = np.unique(batch_indices.astype(int))
        vec = np.empty((n_pts, 2))
        masks = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            to_compute = []
            for b in unique_batch_indices:
                mask = batch_indices == b
                masks.append(mask)
                to_compute.append(
                    executor.submit(
                        self.gridflows[b].query_multiple_labels_threaded,
                        pos=pos[mask, :],
                        labels=labels[mask]
                    )
                )
            for f in concurrent.futures.as_completed(to_compute):
                f: concurrent.futures.Future
                i = to_compute.index(f)
                mask = masks[i]
                vec[mask, :] = f.result()
        return vec
