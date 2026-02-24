import concurrent
import copy
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple

import lightning
import numpy as np
import tifffile
import torch
from natsort import natsorted
from torch import nn
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision.transforms import InterpolationMode, v2
from tqdm import tqdm

from crossgoose.cellpose.transforms import get_pad_yx, normalize99
from crossgoose.gridflow import GridFlow
from crossgoose.mask_utils import convert_labels_to_onehot
from crossgoose.utils import default, imread, remap

ImageNormalization = Literal['M1P1', 'N99']


def normalize_image(image: np.ndarray, image_normalization: ImageNormalization) -> np.ndarray:
    if image_normalization == 'M1P1':
        return remap(image, -1, 1)
    elif image_normalization == 'N99':
        return normalize99(image)
    else:
        raise ValueError


@dataclass
class AugmentationParams():
    patch_size: int | None
    flip_h: bool
    flip_v: bool
    rotate: bool
    rot90: bool
    scale_range: Tuple[float, float] | None
    deterministic_patch: bool = False


def random_transform(
    image: torch.Tensor,
    labels: torch.Tensor,
    overlap_mask: torch.Tensor,
    flows: GridFlow,
    patch_size: int | None,
    flip_h: bool,
    flip_v: bool,
    rotate: bool,
    rot90: bool,
    scale_range: Tuple[float, float] | None = None,
    v_fill_img: float = -1,
    deterministic_patch: bool = False,
    relabel: bool = True,
):
    # expects no bach dim
    transforms = dict()
    assert len(image.shape) == 2

    if scale_range is not None or rotate:
        if scale_range is not None:
            scale = torch.rand(size=(1,)) * \
                (scale_range[1]-scale_range[0]) + scale_range[0]
        else:
            scale = 1.0
        if rotate:
            angle = float(torch.rand(size=(1,)) * 360 - 180)
        else:
            angle = 0.0
        affine_transform = {
            'scale': scale,
            'angle': angle,
            'translate': (0, 0),
            'shear': (0, 0)
        }
        h, w = image.shape
        center = (h/2, w/2)
        image = v2.functional.affine(
            image.unsqueeze(dim=0),
            **affine_transform,
            interpolation=InterpolationMode.BILINEAR,
            fill=v_fill_img
        )[0]
        labels = v2.functional.affine(
            labels.unsqueeze(dim=0),
            **affine_transform,
            interpolation=InterpolationMode.NEAREST,
            fill=0
        )[0].long()
        if overlap_mask is not None:
            overlap_mask = v2.functional.affine(
                overlap_mask.unsqueeze(dim=0),
                **affine_transform,
                interpolation=InterpolationMode.NEAREST,
                fill=0
            )[0].long()

        flows = flows.affine_transform(
            **affine_transform,
            center=center
        )

        transforms['affine'] = affine_transform

    if patch_size is None:
        h, w = image.shape
        ypad1, ypad2, xpad1, xpad2 = get_pad_yx(
            h, w, div=16, extra=1, min_size=None)
        patch_size = max(h+xpad1+xpad2, w+ypad1+ypad2)

    if any(s < patch_size for s in image.shape):
        # print(f"shape {tuple(image.shape)} too small for patch size {patch_size}")
        d0 = max(0, patch_size - image.shape[0])
        d1 = max(0, patch_size - image.shape[1])
        padding = (
            d1 // 2, d1 - d1//2, d0 // 2, d0 - d0//2
        )
        image = nn.functional.pad(image, padding, value=v_fill_img)
        labels = nn.functional.pad(labels, padding, value=0)
        if overlap_mask is not None:
            overlap_mask = nn.functional.pad(overlap_mask, padding, value=0)

        flows = flows.pad(padding)

        assert len(image.shape) == 2

        assert all(
            s >= patch_size for s in image.shape), "oops messed up the padding"

    if deterministic_patch:
        i = image.shape[0]//2 - patch_size//2
        j = image.shape[1]//2 - patch_size//2
        h = patch_size
        w = patch_size
    else:
        i, j, h, w = v2.RandomCrop.get_params(
            image,
            output_size=(patch_size, patch_size)
        )

    transforms['crop'] = (i, j, h, w)

    image = v2.functional.crop(image, i, j, h, w)
    labels = v2.functional.crop(labels, i, j, h, w)
    if overlap_mask is not None:
        overlap_mask = v2.functional.crop(overlap_mask, i, j, h, w)

    flows = flows.crop(i, j, h, w)

    if flip_h and (torch.rand(1) > 0.5):
        image = v2.functional.hflip(image)
        labels = v2.functional.hflip(labels)
        if overlap_mask is not None:
            overlap_mask = v2.functional.hflip(overlap_mask)
        flows = flows.flip(mode='h', shape=image.shape)
        transforms['flip_h'] = True
    else:
        transforms['flip_h'] = False

    if flip_v and (torch.rand(1) > 0.5):
        image = v2.functional.vflip(image)
        labels = v2.functional.vflip(labels)
        if overlap_mask is not None:
            overlap_mask = v2.functional.vflip(overlap_mask)
        flows = flows.flip(mode='v', shape=image.shape)
        transforms['flip_v'] = True
    else:
        transforms['flip_v'] = False

    if rot90:
        k = int(torch.randint(0, 4, size=(1,)))
        image = torch.rot90(image, k=k, dims=(0, 1))
        labels = torch.rot90(labels, k=k, dims=(0, 1))
        if overlap_mask is not None:
            overlap_mask = torch.rot90(overlap_mask, k=k, dims=(0, 1))
        h, w = image.shape
        flows = flows.affine_transform(
            angle=-k*90, center=(h/2, w/2)
        )
        transforms['rot'] = k

    if relabel:
        labels_new_to_old = {}
        current_labels = torch.unique(labels, sorted=True).cpu().numpy()
        current_labels = natsorted(list(set(current_labels) - set([0])))

        new_labels = torch.zeros_like(labels)

        for i, l in enumerate(current_labels):
            new_l = i+1
            new_labels[labels == l] = new_l
            labels_new_to_old[new_l] = l

        labels = new_labels
        flows.relabel(labels_new_to_old)

        transforms['labels_new_to_old'] = labels_new_to_old
    else:
        transforms['labels_new_to_old'] = None

    return image, labels, overlap_mask, flows, transforms


class FlowDataset(Dataset):

    def __init__(
            self,
            data_dir,
            subset,
            augmentation_params: AugmentationParams,
            recompute_flows: bool,
            center_method: str,
            cuda_flow_compute: bool = True,
            closure_radius: int | None = None,
            bootstrap_factor: int = 1,
            lazy_flow_computing: bool = False,
            return_overlap_map: bool = False,
            keep_data_in_memory: bool = False,
            image_normalization: ImageNormalization = 'M1P1',
            gridflow_n_interpol: int = 21
    ):
        self.aug_params = augmentation_params
        self.closure_radius = closure_radius
        self.bootstrap_factor = bootstrap_factor
        self.lazy_flow_computing = lazy_flow_computing
        self.center_method = center_method
        self.cuda_flow_compute = cuda_flow_compute
        self.return_overlap_map = return_overlap_map
        self.keep_data_in_memory = keep_data_in_memory
        self.image_normalization = image_normalization
        self.gridflow_n_interpol = gridflow_n_interpol

        self.name = f"{os.path.split(data_dir)[-1]}-{subset}"

        directory = os.path.join(data_dir, subset)
        directory = Path(directory)

        self.directory = directory

        self._fetch_files()

        assert len(self.images_files) > 0
        assert len(self.images_files) == len(self.masks_files)
        if not len(self.images_files) == len(self.masks_onehot_files):
            print(f"[{self.name}] missing onehot masks: generating ...")
            self.generate_one_hot_masks()
            self._fetch_files()
            assert len(self.images_files) == len(
                self.masks_onehot_files), (self.images_files, self.masks_onehot_files)

        self.flow_files = []
        desc = f"[{self.name}] " + f"checking flows for {subset} data"
        with concurrent.futures.ProcessPoolExecutor() as executor:
            to_compute = []
            for index, filename in enumerate(tqdm(self.images_files, desc=desc)):
                m = re.match(r'(.*)\_img\.(tif|png)', filename)
                assert m is not None, f"image file {filename} does not match pattern .*_img.tif"
                image_name = m.group(1)
                flow_file = f"{image_name}_flows_{center_method}_c{default(closure_radius, 0)}.npz"
                flow_file_path = os.path.join(self.directory, flow_file)
                compute_flow = ((not os.path.exists(flow_file_path))
                                and (not self.lazy_flow_computing))
                compute_flow = compute_flow or recompute_flows
                if compute_flow:
                    to_compute.append(
                        executor.submit(
                            self.compute_flow,
                            flow_file_path=flow_file_path,
                            image_index=index,
                            center_method=center_method,
                            cuda_flow_compute=cuda_flow_compute
                        )
                    )
                self.flow_files.append(flow_file)
            if len(to_compute) > 0:
                desc = f"[{self.name}] Computing flows for {subset} data"
                pbar = tqdm(total=len(to_compute), desc=desc)
                for _ in concurrent.futures.as_completed(to_compute):
                    pbar.update(1)
            else:
                print(f"[{self.name}] no flows to (re)compute {subset} data")

        self.n_images = len(self.images_files)

        if self.keep_data_in_memory:
            self.buffer = {}
        else:
            self.buffer = None

    def _fetch_files(self):

        dir_file = os.listdir(self.directory)

        self.images_files = natsorted(
            [f for f in dir_file if re.search(r'.*\_img\.(tif|png)', f)])
        self.masks_files = natsorted(
            [f for f in dir_file if re.search(r'.*\_masks\.(tif|png)', f)])
        self.masks_onehot_files = natsorted(
            [f for f in dir_file if re.search(r'.*\_masks_onehot\.(tif|png)', f)])

    def generate_one_hot_masks(self):
        for masks_file in tqdm(self.masks_files, desc=f'computing onehot masks for {self.name}'):
            name, _ = masks_file.split('.')
            file = os.path.join(self.directory, f"{name}_onehot.tif")
            if not os.path.exists(file):
                labels = imread(os.path.join(
                    self.directory, masks_file))
                mask_onehot = convert_labels_to_onehot(
                    labels, closure_radius=self.closure_radius)

                tifffile.imwrite(
                    file,
                    data=mask_onehot,
                    compression='zlib',
                    metadata={'axes': 'ZYX'},
                )

    def compute_flow(
        self,
        flow_file_path: str,
        image_index: int,
        center_method: str,
        cuda_flow_compute: bool
    ):

        labels_one_hot = imread(os.path.join(
            self.directory, self.masks_onehot_files[image_index]))

        gf = GridFlow.from_one_hot(
            labels_one_hot=labels_one_hot,
            n_interpol=1,
            flow_center_method=center_method,
            flow_compute_device=torch.device(
                'cuda' if cuda_flow_compute else 'cpu')
        )
        gf.to_file(flow_file_path)

    def __len__(self):
        return self.n_images * self.bootstrap_factor

    def _get_raw_item(self, index):
        if self.keep_data_in_memory and index in self.buffer:
            data = self.buffer[index]
        else:
            data = {}
            image = imread(os.path.join(
                self.directory, self.images_files[index]))
            if len(image.shape) == 3:
                chan_dim = np.argmin(image.shape)
                image = np.mean(image, axis=chan_dim)
            image = self.norm_image(image)
            data['image'] = torch.tensor(image)

            labels = imread(os.path.join(
                self.directory, self.masks_files[index]))
            data['labels'] = torch.tensor(labels, dtype=torch.long)

            if self.keep_data_in_memory:
                self.buffer[index] = data

            flow_file = self.flow_files[index]
            flow_file_path = os.path.join(self.directory, flow_file)
            if flow_file_path is None and self.lazy_flow_computing:
                flow_file_path = os.path.join(self.directory, flow_file)
                print(
                    f"[{self.name}] i'm lazy (or the file is missing), i'm just now computing the flow for image {index}, hold on a sec...")
                self.compute_flow(flow_file_path, index,
                                  self.center_method,
                                  self.cuda_flow_compute)
            else:
                assert os.path.exists(flow_file_path)

            gridflow = GridFlow.from_file(
                flow_file_path, n_interpol=self.gridflow_n_interpol
            )
            data['flows'] = gridflow

            # sanity check
            if np.max(labels) != np.max(gridflow.get_label_keys()):
                raise ValueError(
                    f"got max label {np.max(labels)} and {np.max(gridflow.get_label_keys())} flow slices")

            if self.return_overlap_map:
                labels_oh = imread(os.path.join(self.directory,
                                                self.masks_onehot_files[index]))
                overlap_mask = np.sum(labels_oh, axis=0) > 1
                overlap_mask = torch.from_numpy(overlap_mask)
                assert overlap_mask.shape == labels.shape
            else:
                overlap_mask = None
            data['overlap_mask'] = overlap_mask
        return data

    def norm_image(self, image: np.ndarray) -> np.ndarray:
        return normalize_image(image, self.image_normalization)

    def __getitem__(self, index):
        index = index % self.n_images

        data = self._get_raw_item(index)

        if self.image_normalization == 'M1P1':
            v_fill_img = -1
        elif self.image_normalization == 'N99':
            v_fill_img = 0.0
        else:
            v_fill_img = 0.0

        image, labels, overlap_mask, flows, transforms = random_transform(
            data['image'], data['labels'],
            data['overlap_mask'], data['flows'],
            v_fill_img=v_fill_img,
            **self.aug_params.__dict__
        )

        # nb_instances = int(flows.shape[0])
        ret = {
            'image': image.unsqueeze(dim=0).float(),
            'labels': labels,
            'flows': flows,
            # 'nb_instances': nb_instances, #TODO remove
            'source': self.images_files[index],
            'transforms': transforms
        }
        if self.return_overlap_map:
            ret['overlap_mask'] = overlap_mask
        return ret


def collate_tensor_pad_to_size(batch: List[torch.Tensor], value=0) -> torch.Tensor:
    """collates tensors by padding the first dim to be the same

    Args:
        batch (List[torch.Tensor]): tensors to collate
        value (_type_, optional): padding value. Defaults to 0.

    Returns:
        torch.Tensor: stacked tensors with padding
    """
    elem = batch[0]
    assert isinstance(elem, torch.Tensor)
    # expects tensors of shape n,c,h,w
    n_max = max([e.shape[0] for e in batch])
    base_pad = (0, 0)*3
    return torch.stack([
        nn.functional.pad(
            input=e,
            pad=base_pad + (0, n_max - e.shape[0]),
            value=value
        ) for e in batch
    ], dim=0)


def flow_data_collate_fn(batch):
    elem = batch[0]
    if isinstance(elem, dict):
        clone = copy.copy(elem)
        for key in elem:
            if key == 'flows':
                clone[key] = [d[key] for d in batch]
            elif key == 'transforms':
                clone[key] = [d[key] for d in batch]
            else:
                clone[key] = default_collate([d[key] for d in batch])
        return clone
    else:
        return default_collate(batch)


class MultiDataset(Dataset):
    def __init__(self, data_dir: List[str], **kwargs):
        self.datasets = [
            FlowDataset(data_dir=d, **kwargs) for d in data_dir
        ]
        self.sizes = [d.__len__() for d in self.datasets]
        self.sizes_cumsum = np.cumsum(self.sizes)
        self.offset = np.concatenate(
            (np.array([0]), self.sizes_cumsum), axis=0)

    def __len__(self):
        return sum(self.sizes)

    def __getitem__(self, index):
        d_idx = np.argmax(index < self.sizes_cumsum)
        return self.datasets[d_idx].__getitem__(index=index-self.offset[d_idx])


class FlowDataModule(lightning.LightningDataModule):
    def __init__(
            self,
            data_root: str,
            dataset: str | List[str],
            batch_size: int,
            num_workers: int,
            augmentation_params: AugmentationParams,
            validation: bool = False,
            recompute_flows: bool = False,
            closure_radius: int | None = 8,
            cuda_flow_compute: bool = True,
            center_method: str = 'dist',
            bootstrap_factor: int = 1,
            prefetch_factor: int = 2,
            lazy_flow_computing: bool = False,
            return_overlap_map: bool = False,
            keep_data_in_memory: bool = True,
            val_batch_size: int | None = None,
            val_patch_size: int | None = None,
            image_normalization: ImageNormalization = 'M1P1'
    ):
        super().__init__()
        data_dir = os.path.join(data_root, dataset)

        self.data_dir = data_dir
        self.prefetch_factor = prefetch_factor
        self.train_data, self.test_data = None, None
        self.val_data = None
        self.batch_size = batch_size
        self.val_batch_size = default(val_batch_size, batch_size)
        self.num_workers = num_workers
        self.validation = validation

        self.dataset_params_train = dict(
            augmentation_params=augmentation_params,
            recompute_flows=recompute_flows,
            closure_radius=closure_radius,
            cuda_flow_compute=cuda_flow_compute,
            center_method=center_method,
            bootstrap_factor=bootstrap_factor,
            lazy_flow_computing=lazy_flow_computing,
            return_overlap_map=return_overlap_map,
            keep_data_in_memory=keep_data_in_memory,
            image_normalization=image_normalization
        )
        self.dataset_params_val = dict(
            augmentation_params=AugmentationParams(
                patch_size=default(
                    val_patch_size, augmentation_params.patch_size),
                flip_h=False,
                flip_v=False,
                rotate=False,
                scale_range=None,
                rot90=None,
                deterministic_patch=True
            ),
            bootstrap_factor=1,
            recompute_flows=recompute_flows,
            closure_radius=closure_radius,
            cuda_flow_compute=cuda_flow_compute,
            center_method=center_method,
            lazy_flow_computing=lazy_flow_computing,
            return_overlap_map=return_overlap_map,
            keep_data_in_memory=keep_data_in_memory,
            image_normalization=image_normalization,
        )

    def setup(self, stage):
        if isinstance(self.data_dir, list):
            data_class = MultiDataset
        else:
            data_class = FlowDataset

        self.train_data = data_class(
            subset="train",
            data_dir=self.data_dir,
            **self.dataset_params_train
        )
        self.test_data = data_class(
            subset="test",
            data_dir=self.data_dir,
            **self.dataset_params_val
        )

        if self.validation:
            self.val_data = data_class(
                subset="val",
                data_dir=self.data_dir,
                **self.dataset_params_val
            )

    def val_dataloader(self):
        if self.validation is not None:
            return DataLoader(
                self.val_data,
                batch_size=self.val_batch_size,
                num_workers=self.num_workers,
                # collate_fn=flow_data_collate_fn,#TODO REmove ?
                shuffle=False,
                prefetch_factor=self.prefetch_factor
            )
        else:
            return None

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=flow_data_collate_fn,
            shuffle=True,
            prefetch_factor=self.prefetch_factor
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            collate_fn=flow_data_collate_fn,
            shuffle=False,
            prefetch_factor=self.prefetch_factor
        )
