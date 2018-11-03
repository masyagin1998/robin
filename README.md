# robin

<img src="static/logo/robin.png" height="150" width="150">

**robin** is a **RO**bust document image **BIN**arization tool, written in Python.

- **robin** - fast document image binarization tool;
- **metrics** - script for measuring the quality of binarization;
- **dataset** - links for DIBCO 2009-2018, Palm Leaf Manuscript and my own datasets with original and groun-truth images; script for creating training data from datasets;
- **articles** - selected binarization articles, that helped me a lot;

## Installation

**robin** requires [Python](https://www.python.org/) v3.5+ to run.

Get **robin**, install the dependencies from requirements.txt, download datasets and weights, and now You are ready to binarize documents!

```sh
$ git clone https://github.com/masyagin1998/robin.git
$ cd robin
```
## HowTo

#### Robin

**robin** consists of three files: `src/unet/train.py`, which generates weights for U-net model from input 128x128 pairs of
original and ground-truth images, `src/unet/binarize.py` for binarization group of input document images and `src/unet/model/unet.py` - U-net architecture. Model works with 128x128 images, so binarization tool firstly splits input imags to 128x128 pieces. You can easily rewrite code for different size of U-net image, but researches show that 128 x 128 is the best size.

#### Metrics

You should know, how good is your binarization tool, so I made a script that automates calculation of four DIBCO metrics: F-measure, pseudo F-measure, PSNR and DRD: `src/metrics/metrics.py`. Unfortunately it requires two DIBCO tools: `weights.exe` and `metrics.exe`, which could be started only on Windows (I tried to run them on Linux with Wine, but couldn't, because one of their dependecies is `matlab MCR 9.0 exe`).

#### Dataset

It is realy hard to find good document binarization dataset (DBD), so here I give links to 3 datasets, marked up in a single convenient format. All input image names satisfy `[\d]*_in.png` regexp, and all ground-truth image names satisfy `[\d]*_gt.png` regexp.

- [**DIBCO**](https://yadi.sk/d/_91feeU21y3riA) - 2009 - 2018 competition datasets;
- [**Palm Leaf Manuscript**](https://yadi.sk/d/sMJxS3IGyTRJEA) - Palm Leaf Manuscript dataset from ICHFR2016 competition;
- [**Improved LRDE**](https://yadi.sk/d/-VzpQaQ40Wal9Q) - LRDE 2013 magazines dataset. I improved its ground-truths for better usage;

Also I have simple script - `src/dataset/dataset.py`. It can fastly generate train-validation-testing data from provided
datasets.

#### Articles

While I was working on **robin**, I constantly read some scientific articles. Here I give links to all of them.

- [**DIBCO**](https://yadi.sk/d/2AQHWU0eFsyMvA) - 2009 - 2018 competition articles (I don't have 2017 article, if You have, please, contact me);
- [**DIBCO metrics**](https://yadi.sk/d/fO3KN21inP662g) - articles about 2 non-standard DIBCO metrics: pseudo F-Measure and DRD (PSNR and F-Measure is realy easy to find on the Web);
- [**U-net**](https://yadi.sk/i/5NligqxNbUPCYA) - articles about U-net convolutional network architecture; 
- [**CTPN**](https://yadi.sk/i/oiPxuN_a2a02Eg) - articles about CTPN - fast neural network for finding text in images;
