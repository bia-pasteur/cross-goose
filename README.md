# <img src="goose-logo-rd.png" width="50" height="50" valign="middle"/> Cross-GOOSe

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)]() [![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c?logo=pytorch)]() [![Lightning](https://img.shields.io/badge/Lightning-792ee5)]() [![HAL](https://img.shields.io/badge/HAL-hal--05614392-blue)](https://hal.science/hal-05614392) [![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

_**Cross**ing **G**radient-flows for **O**verlapping **O**bjects **Se**gmentation_

Implementation of _Improving Gradient Flow methods for instance segmentation of crossing objects_ by J. Mabon & J.C. Olivo-Marin, submitted to ISBI 2026.

## 📄 Citation

```bibtex
@inproceedings{mabon2026crossgoose,
  author = {Mabon, J. and Olivo-Marin, J. C.},
  title = {Improving Gradient Flow Methods for Instance Segmentation of Crossing Objects},
  booktitle = {2026 IEEE International Symposium on Biomedical Imaging},
  address = {London, United Kingdom},
  month = {April},
  year = {2026}
}
```

[📄 Paper on HAL](https://hal.science/hal-05614392)


## 👩🏻‍💻 Installation

Setup the environment with [conda/mamba](https://github.com/conda-forge/miniforge):

```bash
mamba create -f env.yaml -y
mamba activate crossgoose
```

## 🏋🏼‍♀️ Training

### Preparing the data

```bash
# Get the data
mkdir -p data/BBBC010_v2_images 
wget "https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v2_images.zip" -O images.zip
unzip images.zip -d data/BBBC010_v2_images
rm images.zip

wget "https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground_eachworm.zip" -O labels.zip
unzip labels.zip -d data
rm labels.zip

# Make a dataset
python main.py make_dataset --config configs/dataset.yaml

# Make a dataset with synthetic data
python main.py make_synth_dataset --config configs/synth_dataset.yaml
```

### Train

```bash
python train.py fit --config configs/model.yaml
```

### Infer/eval

```bash
python main.py eval_models --config configs/eval.yaml
```

## 🌟 Changelog

### v1.3.5
- **Trajectory-based training**: new `TrajectorySampler` and `train_on_trajectories` option to learn from full point trajectories instead of just (u0, ut) pairs
- **Time-error weighting**: `time_error_weighting` option to balance loss contribution across time steps
- **CPLike flow function**: new `FlowFunction` variant that uses only `et` embedding (CellPose-like)
- **Improved GridFlow**:
  - new `from_labels()` method for direct label array loading
  - `keep_largest_component()` to handle non-contiguous masks
  - threaded `query_multiple_labels_threaded()` for faster queries
- **2-channel image support**: model now handles both grayscale and multi-channel images (`grayscale` flag in dataset)
- **Enhanced point sampling**: `RandomOnCellV2` with improved random sampling within instances
- **Augmentation updates**: fixed patching and transforms for multi-channel images
- **New configs**: multiple versioned configs (1.3.0 through 1.3.5) for reproducibility

### v1.2.1
- **Shared embedding architecture**: new `shared_embedding` option to use the same embedding for both u0 and ut, reducing model parameters
- **Embedding visualization**: added notebook and utilities to view learned embeddings
- **Model configuration**: added `model_sharedemb.yaml` config for shared embedding training

### v1.1.1
- **GridFlow optimization**: major refactor of flow computation with `BatchGridFlow` for faster precomputing
- **Data pipeline refactor**: moved sampling to dataloader, reorganized `crossgoose/data/` module structure
- **Threading improvements**: added threaded point sampling for faster data loading
- **Configuration updates**: new model config system (`model2.yaml`), updated defaults
- **Bug fixes**: fixed normalization vector, gridflow augmentation, and removed CUDA flow compute
