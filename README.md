# Rice Biomass CNN

Rice Biomass CNN is a model to estimate the rice above ground biomass (biomass) based on RGB image of rice canopy. The model is developed based on more than 9,000 images of 28 cultivars.
This project is the implementation of the paper "Biomass estimation of World Rice (<i>Oryza sativa</i> L.) Core Collection based on the convolutional neural network and digital images of canopy".

## Performance 

The model explained approximately 95% of variation in observed rice biomass using the test dataset, and 87% of variation using the independent prediction dataset. 

## Conditions on estimation

RGB images that were captured vertically downwards over the rice canopy from 1.5m above the ground using a digital camera should be input. 

![example](https://github.com/KotaNakajima/rice_biomass_CNN/blob/main/example/1.jpg)

## Environment on experiments

### OS

- Ubuntu 18.04.5 LTS

### CPU

- Intel(R) Xeon(R) W-2295 CPU @ 3.00GHz 18 cores

### GPU

- NVIDIA GeForce RTX 3090 x2

### CUDA

- Cuda compilation tools, release 11.3, V11.3.109

### Python

- Python 3.8.8


## Installation

1. Install depentencies.

```bash
pip install -r requirements.txt
```

2. Install Tensorflow

Please install Tensorflow version compatible with your cuda version.

3. Download pre-trained model from google drive.

```bash
mkdir checkpoints
wget "https://drive.google.com/file/d/1GucC686zyi9PYXocQCNUmW43PlmCOW1N/view?usp=sharing" -O rice_biomass_CNN_weights.hdf5
```

## Estimation

Run

```bash
python estimate.py --checkpoint_path checkpoints/rice_biomass_CNN.hdf5 --image_dir example --csv
```

You can find estimated biomass on your console.

Belows are meanings of options.

- checkpoint_path : Path to the checkpoint file you saved.

- image_dir : path to the directory where images are saved.

- csv: If you set this, csv of results will be generated.
