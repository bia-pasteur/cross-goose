"""A compact way to store and query flow fields for overlapping object labels.

GridFlow provides efficient storage of flow vectors at sparse point locations
per label, with interpolation for querying arbitrary positions. Supports
augmentation operations (rotation, scaling, flip, crop, pad) and batched queries.
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
    """Compute flow field for a single-instance mask using the CellPose approach.

    Extends center values outward to generate a potential field, then computes
    gradients to obtain flow vectors pointing toward the center.

    Args:
        mask: Binary mask (H, W) with values 0 or 1.
        center_method: Method to compute the center point ('centroid', 'medoid', etc.).
        n_iter: Number of extension iterations. If None, defaults to 2*(H+W).

    Returns:
        Flow field array of shape (2, H, W) with normalized flow vectors.
    """
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
    """Keep only the largest connected component in a mask.

    Args:
        mask: Binary or integer label mask.

    Returns:
        Binary mask containing only the largest connected component.
    """
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
    """Extract points and flows from a mask by sampling contour and interior.

    Computes a flow field for the mask, then samples points along the contour
    and in the interior at specified densities.

    Args:
        mask: Binary mask (H, W).
        flow_center_method: Method to compute flow center.
        inside_sub_sampling: Subsampling factor for interior points.
        contour_sub_sampling: Subsampling factor for contour points.
        contour_method: 'marching_squares' for precise contours or 'dilation'.

    Returns:
        Tuple of (points, flows) where:
            - points: (N, 2) array of (y, x) coordinates.
            - flows: (N, 2) array of flow vectors at each point.
    """
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
    """Sparse flow field representation per label using KD-trees for interpolation.

    Stores flow vectors at discrete point locations for each label, enabling
    efficient queries at arbitrary positions via k-nearest neighbor interpolation.
    Supports augmentation operations and serialization.
    """
    def __init__(
        self,
        points: Dict[str, np.ndarray],
        flows: Dict[str, np.ndarray],
        n_interpol: int
    ):
        """Initialize GridFlow with points and flows for each label.

        Args:
            points: Dictionary mapping label IDs to (N, 2) point coordinate arrays.
            flows: Dictionary mapping label IDs to (N, 2) flow vector arrays.
            n_interpol: Number of neighbors to use for interpolation.
        """
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
        """Construct GridFlow from one-hot encoded labels.

        Args:
            labels_one_hot: One-hot mask array (K, H, W) where K is the number of labels.
            n_interpol: Number of neighbors for interpolation.
            flow_center_method: Method to compute flow centers.
            inside_sub_sampling: Subsampling factor for interior points.
            contour_sub_sampling: Subsampling factor for contour points.
            contour_method: 'marching_squares' for precise contours or 'dilation'.

        Returns:
            GridFlow instance with flows computed for each label.
        """
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
        """Construct GridFlow from integer label mask.

        Args:
            labels: Integer label mask (H, W) where 0 is background.
            n_interpol: Number of neighbors for interpolation.
            flow_center_method: Method to compute flow centers.
            inside_sub_sampling: Subsampling factor for interior points.
            contour_sub_sampling: Subsampling factor for contour points.
            contour_method: 'marching_squares' for precise contours or 'dilation'.

        Returns:
            GridFlow instance with flows computed for each label.
        """
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
        """Save GridFlow to a compressed .npz file.

        Args:
            file: Output path with .npz extension.
        """
        file = Path(file)
        assert file.suffix == '.npz'

        points = {k: v.data.copy() for k, v in self.points.items()}
        flows = {k: v.copy() for k, v in self.flows.items()}

        data_flat = {f'points/{k}': v for k, v in points.items()}
        data_flat.update({f'flows/{k}': v for k, v in flows.items()})

        np.savez_compressed(file, **data_flat, allow_pickle=False)

    @classmethod
    def from_file(self, file: Path, n_interpol) -> Self:
        """Load GridFlow from a .npz file.

        Args:
            file: Path to .npz file containing points and flows arrays.
            n_interpol: Number of neighbors for interpolation.

        Returns:
            GridFlow instance loaded from file.
        """
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
        """Query flows at multiple positions for multiple labels.

        Args:
            pos: Query positions (N, 2) as (y, x) coordinates.
            labels: Label ID for each position (N,).

        Returns:
            Interpolated flow vectors (N, 2).
        """
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
        """Query flows at multiple positions for multiple labels using threading.

        Parallelizes queries across unique labels for improved performance.

        Args:
            pos: Query positions (N, 2) as (y, x) coordinates.
            labels: Label ID for each position (N,).

        Returns:
            Interpolated flow vectors (N, 2).
        """
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
        """Query flow vectors at given positions for a specific label.

        Uses k-nearest neighbor interpolation from stored point locations.

        Args:
            pos: Query positions (N, 2) as (y, x) coordinates.
            label: Label ID to query.

        Returns:
            Interpolated and normalized flow vectors (N, 2).

        Raises:
            KeyError: If label is not present in the GridFlow.
        """
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
        """Query flows on a dense grid for a given label.

        Args:
            label: Label ID to query.
            shape: Output shape (H, W).

        Returns:
            Dense flow field (2, H, W).
        """
        h, w = shape
        pts = np.reshape(np.mgrid[:h, :w], (2, -1)).transpose()
        flow = self.query(pos=pts, label=label).transpose().reshape((2, h, w))
        return flow

    def get_label_keys(self) -> List[int]:
        """Return list of label IDs present in the GridFlow."""
        return list(self.points.keys())

    def affine_transform(
        self,
        center: Tuple[float, float],
        scale: float = 1.0,
        angle: float = 0.0,
        translate: Tuple[float, float] = (0, 0),
        shear: Tuple[float, float] = (0, 0),
    ) -> Self:
        """Apply affine transformation to points and flows.

        Args:
            center: Center of rotation/scaling (cx, cy).
            scale: Scaling factor.
            angle: Rotation angle in degrees.
            translate: Translation vector (tx, ty).
            shear: Shear angles (not implemented).

        Returns:
            New GridFlow with transformed points and flows.

        Raises:
            NotImplementedError: If shear is non-zero.
        """
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
        """Shift all points by a padding offset.

        Args:
            padding: Tuple (top, bottom, left, right).

        Returns:
            New GridFlow with points shifted by (left, top).
        """
        offset = np.array([[padding[2], padding[0]]])
        return GridFlow(
            points={k: offset + v.data for k, v in self.points.items()},
            flows=self.flows,
            n_interpol=self.n_interpol
        )

    def crop(self, top: int, left: int, height: int, width: int, drop_oob: bool = False) -> Self:
        """Crop the GridFlow to a region of interest.

        Args:
            top: Top crop coordinate.
            left: Left crop coordinate.
            height: Height of the cropped region.
            width: Width of the cropped region.
            drop_oob: If True, drop labels with points outside the crop bounds.

        Returns:
            New cropped GridFlow.
        """
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
        """Flip points and flows horizontally or vertically.

        Args:
            mode: 'h' for horizontal flip, 'v' for vertical flip.
            shape: Image shape (H, W) for computing flipped coordinates.

        Returns:
            New flipped GridFlow.

        Raises:
            ValueError: If mode is not 'h' or 'v'.
        """
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
        """Relabel the GridFlow according to a mapping.

        Args:
            mapping: Dictionary mapping old label IDs to new label IDs.
        """
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
    """Compute a 2x3 affine transformation matrix.

    Args:
        translation: Translation vector (tx, ty).
        center: Center point (cx, cy) for rotation/scaling.
        rot: Rotation angle in radians.
        scale: Scaling factor.

    Returns:
        2x3 affine transformation matrix.
    """
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
    """Compute a 2D rotation matrix.

    Args:
        angle: Rotation angle (in radians unless degree=True).
        degree: If True, interpret angle as degrees.

    Returns:
        2x2 rotation matrix.
    """
    if degree:
        angle = math.radians(angle)
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


def _transform_points(x: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """Apply affine transform to point coordinates.

    Args:
        x: Points array (N, 2).
        transform_matrix: 2x3 affine transformation matrix.

    Returns:
        Transformed points (N, 2).
    """
    x = np.matmul(transform_matrix, np.concat(
        [x, np.ones((x.shape[0], 1))], axis=1).T).T
    return x[:, :2]


def _transform_flows(x: np.ndarray, angle: float) -> np.ndarray:
    """Apply rotation to flow vectors (translation and scale have no effect).

    Args:
        x: Flow vectors (N, 2).
        angle: Rotation angle in degrees.

    Returns:
        Rotated flow vectors (N, 2).
    """
    # translate and scale have no effect
    if angle != 0.0:
        rot_matrix = get_rotation_matrix(-angle, degree=True)
        x = np.matmul(rot_matrix, x.transpose()).transpose()

    return x


class BatchGridFlow:
    """Batch container for multiple GridFlow instances supporting parallel queries.

    Wraps a list of GridFlow instances (one per batch element) and provides
    efficient batched queries across labels and batch indices using threading.
    """
    def __init__(
        self,
        gridflows: List[GridFlow]
    ):
        """Initialize BatchGridFlow with a list of GridFlow instances.

        Args:
            gridflows: List of GridFlow instances, one per batch element.
        """
        self.gridflows: List[GridFlow] = gridflows

    def __getitem__(self, key) -> GridFlow:
        """Get a GridFlow by index."""
        return self.gridflows[key]

    def batch_query_multiple_labels(
        self,
        pos: np.ndarray,
        batch_indices: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Query flows for multiple positions across multiple batch elements.

        Parallelizes queries across batch indices using threading for efficiency.

        Args:
            pos: Query positions (N, 2) as (y, x) coordinates.
            batch_indices: Batch index for each position (N,).
            labels: Label ID for each position (N,).

        Returns:
            Interpolated flow vectors (N, 2).
        """
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
