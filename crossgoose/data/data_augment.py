from dataclasses import dataclass

from crossgoose.cellpose.transforms import get_pad_yx
from crossgoose.gridflow import GridFlow


import torch
from natsort import natsorted
from torch import nn
from torchvision.transforms import InterpolationMode, v2


from typing import Tuple


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


@dataclass
class AugmentationParams():
    patch_size: int | None
    flip_h: bool
    flip_v: bool
    rotate: bool
    rot90: bool
    scale_range: Tuple[float, float] | None
    deterministic_patch: bool = False