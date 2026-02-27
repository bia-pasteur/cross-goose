import concurrent
import copy
import os
import re
from pathlib import Path
from typing import List

import lightning
import numpy as np
import tifffile
import torch
from natsort import natsorted
from torch import nn
from torch.utils.data import DataLoader, Dataset, default_collate
from tqdm import tqdm

from crossgoose.data.data_augment import AugmentationParams, random_transform
from crossgoose.data.points_sampling import PointsSamlper
from crossgoose.gridflow import BatchGridFlow, GridFlow
from crossgoose.mask_utils import convert_labels_to_onehot
from crossgoose.utils import ImageNormalization, default, imread, normalize_image


class FlowDataset(Dataset):

    def __init__(
            self,
            data_dir,
            subset,
            augmentation_params: AugmentationParams,
            points_sampler: PointsSamlper,
            recompute_flows: bool,
            center_method: str,
            closure_radius: int | None = None,
            bootstrap_factor: int = 1,
            lazy_flow_computing: bool = False,
            return_overlap_map: bool = False,
            keep_data_in_memory: bool = False,
            image_normalization: ImageNormalization = 'M1P1',
            gridflow_n_interpol: int = 21,

    ):
        self.aug_params = augmentation_params
        self.points_sampler = points_sampler
        self.closure_radius = closure_radius
        self.bootstrap_factor = bootstrap_factor
        self.lazy_flow_computing = lazy_flow_computing
        self.center_method = center_method
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
                            center_method=center_method
                        )
                    )
                self.flow_files.append(flow_file)
            if len(to_compute) > 0:
                desc = f"[{self.name}] Computing flows for {subset} data"
                pbar = tqdm(total=len(to_compute), desc=desc)
                for f in concurrent.futures.as_completed(to_compute):
                    _ = f.result()
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
        center_method: str
    ) -> str:

        labels_one_hot = imread(os.path.join(
            self.directory, self.masks_onehot_files[image_index]))
        gf = GridFlow.from_one_hot(
            labels_one_hot=labels_one_hot,
            n_interpol=1,
            flow_center_method=center_method
        )
        gf.to_file(flow_file_path)

        return flow_file_path

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
                                  self.center_method)
            else:
                if not os.path.exists(flow_file_path):
                    raise FileNotFoundError(flow_file_path)

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

        image, labels, overlap_mask, grid_flow, transforms = random_transform(
            data['image'], data['labels'],
            data['overlap_mask'], data['flows'],
            v_fill_img=v_fill_img,
            **self.aug_params.__dict__
        )

        # sample u0, ut, flows
        u0, ut, flow_u0_ut = self.points_sampler.sample(
            image=image,
            labels=labels,
            grid_flow=grid_flow
        )

        ret = {
            'image': image.unsqueeze(dim=0).float(),
            'labels': labels,
            'flowgrid': grid_flow,
            'source': self.images_files[index],
            'transforms': transforms,
            'u0': u0,
            'ut': ut,
            'flow_u0_ut': flow_u0_ut
        }
        if self.return_overlap_map:
            ret['overlap_mask'] = overlap_mask
        return ret


def flow_data_collate_fn(batch):
    elem = batch[0]
    if isinstance(elem, dict):
        clone = copy.copy(elem)
        for key in elem:
            if key == 'flowgrid':
                clone[key] = BatchGridFlow([d[key] for d in batch])
            elif key == 'transforms':
                clone[key] = [d[key] for d in batch]
            elif key in ['u0', 'ut']:
                # prepend batch id in coordinates so that it is (b,i,j)
                clone[key] = torch.concat([
                    torch.concat(
                        [torch.full(
                            size=(d[key].shape[0], 1),
                            fill_value=b,
                            dtype=d[key].dtype,
                            device=d[key].device),
                         d[key]],
                        axis=1)
                    for b, d in enumerate(batch)],
                    axis=0)
            elif key == 'flow_u0_ut':
                clone[key] = torch.concat([d[key] for d in batch], axis=0)
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
            points_sampler: PointsSamlper,
            validation: bool = False,
            recompute_flows: bool = False,
            closure_radius: int | None = 8,
            center_method: str = 'dist',
            bootstrap_factor: int = 1,
            prefetch_factor: int = 2,
            lazy_flow_computing: bool = False,
            return_overlap_map: bool = False,
            keep_data_in_memory: bool = True,
            val_batch_size: int | None = None,
            val_patch_size: int | None = None,
            image_normalization: ImageNormalization = 'M1P1',
            gridflow_n_interpol: int = 21
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
            points_sampler=points_sampler,
            recompute_flows=recompute_flows,
            closure_radius=closure_radius,
            center_method=center_method,
            bootstrap_factor=bootstrap_factor,
            lazy_flow_computing=lazy_flow_computing,
            return_overlap_map=return_overlap_map,
            keep_data_in_memory=keep_data_in_memory,
            image_normalization=image_normalization,
            gridflow_n_interpol=gridflow_n_interpol
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
            points_sampler=points_sampler,  # TODO maybe don't put it there ?
            bootstrap_factor=1,
            recompute_flows=recompute_flows,
            closure_radius=closure_radius,
            center_method=center_method,
            lazy_flow_computing=lazy_flow_computing,
            return_overlap_map=return_overlap_map,
            keep_data_in_memory=keep_data_in_memory,
            image_normalization=image_normalization,
            gridflow_n_interpol=gridflow_n_interpol
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
                collate_fn=flow_data_collate_fn,
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
