# Pytorch implementation of Generalized-One-shot-GAN-Adaption
## 1. Quick start
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A6moUzSLh2vU4CckfnXESk4HwYC8NgMz?usp=sharing)

We provide some [pre-trained models](https://drive.google.com/drive/folders/1xgBY3UyQkR0co_dOfr9SwUy-9h4hGhmc?usp=sharing) and a [inference demo on Google Colab](https://colab.research.google.com/drive/1A6moUzSLh2vU4CckfnXESk4HwYC8NgMz?usp=sharing) to reproduce the qualitative results in paper. It also includes the command to train the model.

## 2. Usage
-  Please refer to the colab to create the envs and download the source models, it mainly includes [StyleGAN](https://github.com/NVlabs/stylegan3) and [CLIP](https://github.com/openai/CLIP) dependencies, and some commonly used packages like cv2.
- We have prepared the scripts to train new models 
  -  For training model without entities, ```sh scripts/train_OSGA.sh```
  -  For training model with entities, ```sh scripts/train_GOGA.sh```
  -  For inference, ```sh scripts/inference.sh```

## 3. Note
This project has not been well cleaned and may contain some redundant files. Please use them sparingly. 

