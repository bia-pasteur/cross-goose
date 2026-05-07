# 🪿 Cross-GOOSe
**Cross**ing **G**radient-flows for **O**verlapping **O**bjects **Se**gmentation

implementation of _Improving Gradient Flow methods for instance segmentation of crossing objects_ J. Mabon & J.C. Olivo-Marin, submitted to ISBI 2026


Cite as:
```
J Mabon, J C Olivo-Marin. Improving Gradient Flow Methods for Instance Segmentation of Crossing Objects. 
2026 IEEE International Symposium on Biomedical Imaging, IEEE, Apr 2026, London, United Kingdom. 
```
[📄 Paper on HAL](https://hal.science/hal-05614392)


## 👩🏻‍💻 Installation
Setup the environment with [conda/mamba](https://github.com/conda-forge/miniforge) :
```bash
mamba create -f env.yaml -y
mamba activate crossgoose
```

## 🏋🏼‍♀️ Training

### Preparing the data
```bash
# get the data
mkdir -p data/BBBC010_v2_images 
wget "https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v2_images.zip" -O images.zip
unzip images.zip -d data/BBBC010_v2_images
rm images.zip

wget "https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground_eachworm.zip" -O labels.zip
unzip labels.zip -d data
rm labels.zip

# make a dataset
python main.py make_dataset --config configs/dataset.yaml
# make a dataset with synthetic data
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

## 🌟 Updates
### v1.1.1
- **GridFlow optimization**: major refactor of flow computation with `BatchGridFlow` for faster precomputing
- **Data pipeline refactor**: moved sampling to dataloader, reorganized `crossgoose/data/` module structure
- **Threading improvements**: added threaded point sampling for faster data loading
- **Configuration updates**: new model config system (`model2.yaml`), updated defaults
- **Bug fixes**: fixed normalization vector, gridflow augmentation, and removed CUDA flow compute

### v1.2.1
- **Shared embedding architecture**: new `shared_embedding` option to use the same embedding for both u0 and ut, reducing model parameters
- **Embedding visualization**: added notebook and utilities to view learned embeddings
- **Model configuration**: added `model_sharedemb.yaml` config for shared embedding training
