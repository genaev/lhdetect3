# LHDetect3

Counting and length measurement of leaf trichomes with an RF-DETR-Seg Small model trained on soybean and wheat leaf-fold images.

Training data: Roboflow project [lhdetect3](https://app.roboflow.com/lhdetect3), dataset `soy_wheat`.

## Model

`weights/rfdetrseg_small_wheat_soy.pth` is RF-DETR-Seg Small (33.4 M parameters, input 1272 × 1272 pixels) trained on the combined soybean + wheat dataset. Of three training runs with different random seeds, this is the run with the highest count F1 on the validation set. The default confidence threshold (0.37) is the value that maximizes count F1 on the validation set.

Accuracy on the test sets at the default threshold:

| Test set | Images | Trichomes | Count F1 | MAE, trichomes per image |
|---|---|---|---|---|
| Soybean | 42 | 1450 | 0.968 | 2.14 |
| Wheat | 29 | 634 | 0.918 | 3.59 |

`infer.py` and the desktop application report the same count: the number of trichomes detected by the model.

## Desktop application

A program with a graphical interface for Windows, Linux and macOS is in [`gui/`](gui/): choose a folder of images, set the scale in micrometers per pixel, and get a table with the number and lengths of trichomes on every image, with the contours of the detected trichomes shown on the images. It uses the GPU automatically when available (NVIDIA, AMD, Intel, Apple).

- Windows: installer in [`gui/dist/windows`](gui/dist/windows/)
- Linux and macOS: `cd gui && ./install.sh` (see [`gui/README.md`](gui/README.md))

## Installation

```bash
git lfs install
git clone https://github.com/genaev/lhdetect3.git
cd lhdetect3
conda create -n lhdetect3 python=3.11
conda activate lhdetect3
pip install -r requirements.txt
```

The weights (about 130 MB) are stored with Git LFS. A CUDA GPU is recommended.

## Usage

```bash
python infer.py                                   # example images in examples/
python infer.py /path/to/images --output results.csv
```

Options: `--conf` confidence threshold (default 0.37), `--gpu` CUDA device (`-1` for CPU), `--weights` checkpoint path. Run `python infer.py -h` for all options.

## Output

One CSV row per image:

```text
image,count,n_measured,lengths_px
wheat/1520_3_1_1a.png,28,28,84.2;91.0;...
```

`count` is the number of detected trichomes; `n_measured` is the number of them whose mask is large enough to measure (at least `--min-area` pixels); `lengths_px` lists their centerline lengths in pixels, separated by semicolons.

## Length measurement

The length of a trichome is the length of the centerline of its mask (`centerline.py`): the longest path through the skeleton of the mask, smoothed (Gaussian, σ = 3 px) and prolonged at both ends to the mask edge; pieces of one mask that continue each other are joined. On the validation set, applied to the manual polygon annotation (2759 trichomes), it reproduces the geodesic length of the polygons with a mean absolute error of 1.1 px (2.7%).
