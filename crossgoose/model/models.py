from __future__ import annotations

import glob
import os
import pathlib
import re
import time
from enum import Enum
from typing import Literal

import lightning
import numpy as np
import torch
import yaml
from lightning.pytorch.cli import LightningArgumentParser
from torch import nn

from crossgoose.cellpose.dynamics import get_masks_torch
from crossgoose.cellpose.metrics import average_precision
from crossgoose.cellpose.resnet_torch import batchconv
from crossgoose.cellpose.transforms import get_pad_yx
from crossgoose.gridflow import BatchGridFlow
from crossgoose.model.flow_function import FlowFunction
from crossgoose.model.modules import CPBackbone

SAVED_MODELS_DIR = pathlib.Path(
    __file__).parent.resolve().joinpath('../saved_models')


class SamplingMethod(Enum):
    FOLLOW_FLOWS = "follow_flows"
    RANDOM_ON_CELL = "random_on_cell"


Ckptcriterion = Literal['last', 'best_ap_0.5', 'best_ap_0.75', 'best_ap_0.9']


class CrossGooseModel(lightning.LightningModule):
    def __init__(
        self,
        flow_fn: FlowFunction,
        embeddings_dim: int = 8,
        crit_flow_weight: float = 0.1,
        crit_cellprob_weight: float = 2.0,
        n_steps: int = 200,
        backbone_realease_delay: int | None = None,
        shared_embedding: bool = False,
        train_on_trajectories: bool = False,
        time_error_weighting: bool = False
    ):
        super().__init__()
        self.n_steps = n_steps
        self.embeddings_dim = embeddings_dim
        self.backbone_realease_delay = backbone_realease_delay
        self.shared_embedding = shared_embedding
        self.train_on_trajectories = train_on_trajectories
        self.time_error_weighting = time_error_weighting

        self.backbone = CPBackbone(device=self.device)
        self.backbone_out = self.backbone.nbase[1]

        self.flow_fn = flow_fn

        if self.shared_embedding:
            out_c = self.embeddings_dim + 1
        else:
            out_c = (2*self.embeddings_dim) + 1

        self.embedding_head = batchconv(
            in_channels=self.backbone_out,
            out_channels=out_c,
            sz=1
        )

        self.criterion_flow = nn.MSELoss(
            reduction="mean" if not self.train_on_trajectories else 'none')
        self.criterion_cellprob = nn.BCEWithLogitsLoss(reduction="mean")

        self.flow_fac = 5.
        self.crit_flow_weight = crit_flow_weight
        self.crit_cellprob_weight = crit_cellprob_weight

        if self.backbone_realease_delay:
            self.backbone.requires_grad_(False)
            print(
                f"[0] freezing backbone training until epoch={self.backbone_realease_delay}")

        self.save_hyperparameters(ignore=['flow_fn', 'positional_embedding'])

    @staticmethod
    def load_model(model: str = 'default', ckpt_crit: Ckptcriterion = 'last') -> CrossGooseModel:
        """loads a model
        Args:
            model (str): path to a directory or name of a default model (eg 'default')
                If path to a directory, it should have format like: 
                    ├── checkpoints
                    │   ├── best-epoch=163-step=181712-val_ap_0.5=0.9320.ckpt
                    │   ├── best-epoch=223-step=248192-val_ap_0.9=0.2635.ckpt
                    │   ├── best-epoch=38-step=43212-val_ap_0.75=0.8247.ckpt
                    │   └── last.ckpt
                    └── config.yaml
            ckpt_crit (Ckptcriterion) : if model is a directory, 
                one of 'last', 'best_ap_0.5', 'best_ap_0.75', 'best_ap_0.9'

        Returns:
            CrossGooseModel: loaded model
        """
        if os.path.isdir(model):
            return CrossGooseModel._load_model_from_dir(model, ckpt_crit=ckpt_crit)
        else:
            if not os.path.exists(SAVED_MODELS_DIR):
                raise FileNotFoundError(
                    f"could not find models dir at {str(SAVED_MODELS_DIR)} or {model} is not a dir")
            available_models = os.listdir(SAVED_MODELS_DIR)
            if not model in available_models:
                raise FileNotFoundError(
                    f"no model {model} in exisitng models {available_models}")

            ckpt_path = os.path.join(SAVED_MODELS_DIR, model, 'weights.ckpt')
            config_path = os.path.join(SAVED_MODELS_DIR, model, 'config.yaml')
            return CrossGooseModel._load_from_ckpt(
                ckpt_path=ckpt_path,
                config_path=config_path
            )

    @staticmethod
    def _load_from_ckpt(
        ckpt_path: str, config_path: str
    ) -> CrossGooseModel:
        assert os.path.exists(ckpt_path)
        assert os.path.exists(config_path)

        parser = LightningArgumentParser()
        parser.add_class_arguments(CrossGooseModel, 'model')
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded_cfg_dict = yaml.safe_load(f)
        cfg_dict = {'model': loaded_cfg_dict['model']}
        loaded_cfg = parser.parse_object(cfg_dict)
        exp = parser.instantiate_classes(loaded_cfg)
        model: CrossGooseModel = exp.model
        with open(ckpt_path, 'rb') as f:
            state_dict = torch.load(f, weights_only=True)
        model.load_state_dict(state_dict['state_dict'])
        print(f"loaded model from {ckpt_path}")
        return model

    @staticmethod
    def _load_model_from_dir(
        model_dir: str,
        ckpt_crit: Ckptcriterion = 'last'
    ) -> CrossGooseModel:
        ckpt_dir = os.path.join(model_dir, 'checkpoints')
        checkpoints = glob.glob(ckpt_dir+'/*.ckpt')
        if len(checkpoints) > 0:
            if ckpt_crit == 'last':
                ckpt = [f for f in checkpoints if re.search(
                    'last', os.path.split(f)[-1])]
                assert len(ckpt) == 1
                ckpt_path = ckpt[0]
            elif 'best_ap' in ckpt_crit:
                if ckpt_crit == 'best_ap_0.5':
                    re_pat = re.compile(r'best-.*-val_ap_0\.5=(.*)\.ckpt')
                elif ckpt_crit == 'best_ap_0.75':
                    re_pat = re.compile(r'best-.*-val_ap_0\.75=(.*)\.ckpt')
                elif ckpt_crit == 'best_ap_0.9':
                    re_pat = re.compile(r'best-.*-val_ap_0\.9=(.*)\.ckpt')
                else:
                    raise ValueError(ckpt_crit)

                files = []
                metrics = []
                for p in checkpoints:
                    filename = os.path.split(p)[-1]
                    m = re_pat.match(filename)
                    if m is not None:
                        files.append(p)
                        metrics.append(float(m.group(1)))

                best_i = np.argmax(metrics)
                ckpt_path = files[best_i]
            else:
                raise ValueError(ckpt_crit)
        else:
            raise FileNotFoundError(f"not *ckpt files at {ckpt_dir}")
        config_path = os.path.join(model_dir, 'config.yaml')
        return CrossGooseModel._load_from_ckpt(
            ckpt_path=ckpt_path,
            config_path=config_path
        )

    def image_to_maps(self, image, apply_sigmoids: bool = False):
        assert len(image.shape) == 4

        T0, _, _ = self.backbone(image)
        T1 = self.embedding_head(T0)

        res = {'T1': T1}

        res['emb_grid_0'] = T1[:, :self.embeddings_dim]
        if self.shared_embedding:
            # share the embedding for u0 and ut
            res['emb_grid_t'] = res['emb_grid_0']
        else:
            res['emb_grid_t'] = T1[:, self.embeddings_dim:2*self.embeddings_dim]
        cp_est = T1[:, -1]
        if apply_sigmoids:
            res['cp_est'] = nn.functional.sigmoid(cp_est)
        else:
            res['cp_est'] = cp_est

        return res

    def forward(self, image: torch.Tensor, u0: torch.Tensor, ut: torch.Tensor):
        _, c, _, _ = image.shape

        if c == 1:
            image = torch.tile(image, (1, 2, 1, 1))

        features = self.image_to_maps(image, apply_sigmoids=True)
        emb_grid_0 = features['emb_grid_0']
        emb_grid_t = features['emb_grid_t']

        e0 = self._gather_emb_batch(
            emb_grid_0, u0)
        et = self._gather_emb_batch(
            emb_grid_t, ut)

        dP = self.flow_fn(e0, et)

        return dP, features['cp_est'], features['T1'], features.get('overlap_est', None)

    def _gather_emb_batch(self, raster: torch.Tensor, u: torch.Tensor):
        # expects raster of shape (B,dim_emb,H,W)
        # TODO decorelate the batch id (u[:, 0]) to the other coordinates
        idx0 = u[..., 0].long()
        idx1 = u[..., 1].long()
        idx2 = u[..., 2].long()
        gathered_emb = raster[idx0, :, idx1, idx2]
        return gathered_emb

    def _sample_gtflows_batch(self, flow_grid_gt: BatchGridFlow, l0: torch.Tensor, ut: torch.Tensor):

        flows = flow_grid_gt.batch_query_multiple_labels(
            pos=ut[..., 1:].detach().cpu().numpy(),
            labels=l0.detach().cpu().numpy(),
            batch_indices=ut[..., 0].detach().cpu().numpy()
        )
        return torch.from_numpy(flows).to(ut.device).float()

    def training_step(self, batch, batch_idx):

        labels: torch.Tensor = batch['labels']
        image: torch.Tensor = torch.tile(batch['image'], (1, 2, 1, 1))

        batch_size = image.shape[0]

        loss_dict = {'loss': 0.0}

        features = self.image_to_maps(image, apply_sigmoids=False)
        emb_grid_0 = features['emb_grid_0']
        emb_grid_t = features['emb_grid_t']

        cell_gt = (labels > 0).float()
        loss_dict['loss_cp'] = self.criterion_cellprob(
            features['cp_est'], cell_gt
        ) * self.crit_cellprob_weight

        loss_dict['loss'] += loss_dict['loss_cp']

        # get points
        pts_coord = batch['pts_coord']
        pts_flows = batch['pts_flows']
        pts_mask = batch['pts_mask']
        pts_batch = batch['pts_batch']

        if self.train_on_trajectories:
            if pts_coord.shape[1] == 2:
                raise print(
                    "WARNING: train_on_trajectories is True but dataloader "
                    f"provided a set of points of length {pts_coord.shape[1]}==2. "
                    "This looks like a two points sampler. "
                    "Try changing the dataloader points_sampler.")

            u0 = torch.concat([pts_batch, pts_coord[:, 0]], axis=-1)
            e0 = self._gather_emb_batch(emb_grid_0, u0)

            n_samples, n_steps, _ = pts_coord.shape

            pts_batch = torch.tile(pts_batch[..., None], (1, n_steps, 1))
            ut = torch.concat([pts_batch, pts_coord], axis=-1)

            et = self._gather_emb_batch(emb_grid_t, ut)

            # tile e0 to et shape
            e0 = torch.tile(e0[:, None], (1, n_steps, 1))

            flow_est = self.flow_fn(e0, et)

            error = torch.mean(self.criterion_flow(
                flow_est, self.flow_fac * pts_flows), dim=-1)  # reduce on spatial dim

            if self.time_error_weighting:
                error_per_timeframe = torch.sum(error, dim=1, keepdim=True)
                error_per_timeframe = error_per_timeframe / \
                    torch.sum(error_per_timeframe)  # normalize
                loss_steps = torch.sum(error * error_per_timeframe) / n_samples
            else:
                loss_steps = torch.mean(error)

        else:
            # v1.0 behaviour
            if pts_coord.shape[1] != 2:
                raise ValueError(
                    "train_on_trajectories is False but dataloader "
                    f"provided a set of points of length {pts_coord.shape[1]}!=2. "
                    "Try changing the dataloader points_sampler.")

            # concat the batch dim
            u0 = torch.concat([pts_batch, pts_coord[:, 0]], axis=-1)
            ut = torch.concat([pts_batch, pts_coord[:, 1]], axis=-1)
            flow_gt = pts_flows[:, 1]

            e0 = self._gather_emb_batch(emb_grid_0, u0)
            et = self._gather_emb_batch(emb_grid_t, ut)

            flow_est = self.flow_fn(e0, et)

            loss_steps = self.criterion_flow(flow_est, self.flow_fac * flow_gt)

        loss_dict['loss_steps'] = loss_steps
        loss_dict['loss'] += loss_dict['loss_steps']

        self.log_dict(
            loss_dict, prog_bar=True,
            logger=True, on_step=False, on_epoch=True,
            batch_size=batch_size
        )

        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        images: torch.Tensor = batch['image']
        batch_size, _, h, w = images.shape
        thresholds = [0.5, 0.75, 0.9]
        masks_true = [batch['labels'][i].squeeze().cpu().numpy()
                      for i in range(batch_size)]
        masks_pred = []
        for i in range(batch_size):
            image = images[[i]]
            results = self.segment_image(image=image)
            masks_pred.append(results['mask'])

        ap, _, _, _ = average_precision(
            masks_true=masks_true,
            masks_pred=masks_pred,
            threshold=thresholds
        )
        log = {f"val_ap_{t}": float(np.nanmean(ap[:, i]))
               for i, t in enumerate(thresholds)}

        self.log_dict(log, batch_size=batch_size, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    @torch.no_grad
    def follow_flow(
        self,
        image,
        n_steps: int,
        as_numpy: bool = True,
        u0: torch.Tensor | None = None,
        skip_logging: bool = False,
    ):

        _, c, h, w = image.shape
        if c == 1:
            image = torch.tile(image, (1, 2, 1, 1))

        features = self.image_to_maps(image, apply_sigmoids=True)
        emb_grid_t = features['emb_grid_t']
        emb_grid_0 = features['emb_grid_0']

        fg = features['cp_est'] > 0.5

        if u0 is None:
            u0 = torch.nonzero(fg)

        min_bound = torch.tensor([0, 0], device=u0.device)
        max_bound = torch.tensor([h, w], device=u0.device)-1

        e0 = self._gather_emb_batch(
            emb_grid_0, u0)

        ut = u0.clone().float()
        log_ut = [u0.clone().float()]
        for _ in range(n_steps):
            et = self._gather_emb_batch(
                emb_grid_t, ut)
            flow_est = self.flow_fn(e0, et) / self.flow_fac
            ut_next = torch.concat((
                ut[:, [0]],
                torch.clamp(
                    ut[:, 1:] + flow_est,
                    min=min_bound, max=max_bound
                )
            ), dim=1)
            if skip_logging:
                log_ut = [ut_next]
            else:
                log_ut.append(ut_next)
            ut = ut_next

        if skip_logging:
            log_ut = log_ut[0]
        else:
            log_ut = torch.stack(log_ut, dim=0)

        if as_numpy:
            log_ut = log_ut.cpu().numpy()
            fg = fg.cpu().numpy()

        return fg, log_ut, features

    @torch.no_grad
    def segment_image(
        self,
        image: torch.Tensor
    ):
        assert len(image.shape) == 4, "expects grayscale images (for now)"
        b, c, h, w = image.shape
        assert b == 1
        assert c in [1, 2]

        timings = {}
        results = {}

        ypad1, ypad2, xpad1, xpad2 = get_pad_yx(
            h, w, div=16, extra=1, min_size=None)

        image_padded = nn.functional.pad(
            image, (xpad1, xpad2, ypad1, ypad2), value=-1)

        # follow flows
        start = time.perf_counter()
        fg, last_pt, features = self.follow_flow(
            image_padded.to(self.device), n_steps=self.n_steps,
            skip_logging=True)
        end = time.perf_counter()
        timings['follow_flow'] = end - start

        cp_est = features['cp_est'].cpu().numpy()[0]
        cp_est = cp_est[ypad1:-ypad2, xpad1:-xpad2]
        results['cellprob'] = cp_est

        # compute masks
        start = time.perf_counter()

        b = 0
        batch_mask = last_pt[:, 0] == b
        batch_pt = torch.from_numpy(
            last_pt[batch_mask][:, 1:]).permute(1, 0).long()
        fgb = fg[b]
        shape0 = fgb.shape
        inds = np.nonzero(fgb)

        masks = get_masks_torch(
            pt=batch_pt,
            inds=inds,
            shape0=shape0
        )

        masks = masks[ypad1:-ypad2, xpad1:-xpad2]
        assert masks.shape == (h, w)

        end = time.perf_counter()
        timings['compute_masks'] = end - start

        results['mask'] = masks
        results['timings'] = timings
        return results

    def on_train_epoch_start(self):
        trainer = self.trainer
        epoch = trainer.current_epoch
        if self.backbone_realease_delay == epoch:
            self.backbone.requires_grad_(True)
            print(f"[{epoch}] releasing backbone training")
