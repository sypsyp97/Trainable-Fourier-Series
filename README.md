# Trainable-Fourier-Series

[![arXiv](https://img.shields.io/badge/arXiv-2401.16039-b31b1b.svg)](https://arxiv.org/abs/2401.16039)


PyTorch implementation of the paper ["Data-Driven Filter Design in FBP: Transforming CT Reconstruction with Trainable Fourier Series"](https://arxiv.org/abs/2401.16039).

## Requirements

The code is developed using Python 3.11 and PyTorch 2.0.1. 
```bash
conda env create -f environment.yml
conda activate tfr
```
Then you need to install [PYRO-NN](pyronn).

## Data
Low-dose CT data can be found [here](https://www.nature.com/articles/s41597-021-00893-z#code-availability).


## Usage Example

```bash
python main.py 
```

## Citation

```
@article{sun2024data,
  title={Data-Driven Filter Design in FBP: Transforming CT Reconstruction with Trainable Fourier Series},
  author={Sun, Yipeng and Schneider, Linda-Sophie and Fan, Fuxin and Thies, Mareike and Gu, Mingxuan and Mei, Siyuan and Zhou, Yuzhong and Bayer, Siming and Maier, Andreas},
  journal={arXiv preprint arXiv:2401.16039},
  year={2024}
}
```