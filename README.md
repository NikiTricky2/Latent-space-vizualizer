# Latent space vizualizer
Explore various latent space representations of the data.

## How to use
```
usage: main.py [-h] [--dims DIMS] [--size SIZE] encoder decoder dataset

Process some integers.

positional arguments:
  encoder      Encoder model file
  decoder      Decoder model file
  dataset      Dataset path

options:
  -h, --help   show this help message and exit
  --dims DIMS  Latent space dimensions
  --size SIZE  Image size
```

## Pretrained models
You can find 2 pretrained model with their datasets in the root folder.
* `cifar10` - Training on CIFAR10 dataset with 32 dimensions and 32x32 image size
* `faces` - Trained on the FFHQ dataset with 16 dimensions and 128x128 image size. **Warning:** The model files are uploaded to HuggingFace due to size limitations. You can see the model [here](https://huggingface.co/NikiTricky/ffhq-autoencoder-16dim)

## Training
You can train your own models using the training.ipynb notebook. The code is made for training on the FFHQ dataset with 16 dimensions and 128x128 image size. You will need to change your model settings with a different dataset.

## Usage
Press on the latent space to see the corresponding image. Change the coordinates to see different perspectives.