# LHDetect3: automated counting and length measurement of leaf trichomes

This repository contains the inference code and the trained model accompanying the article

> Genaev, M.A.; Afonnikov, D.A. *Quantitative assessment of soybean leaf pubescence from RGB images using deep learning and instance segmentation.* Manuscript submitted to *Plants* (MDPI), 2026.

The model detects individual trichomes on microscope images of a leaf fold, counts them and measures the length of each trichome along its centerline. It was trained on soybean (*Glycine max*) and bread wheat (*Triticum aestivum*) images acquired under three imaging protocols.

## Contents

```text
.
├── infer.py              # inference: trichome count and centerline lengths per image
├── centerline.py         # mask post-processing and centerline length measurement
├── requirements.txt      # Python dependencies
├── weights/
│   └── rfdetrseg_small_wheat_soy.pth   # trained model (stored with Git LFS, 129 MB)
└── examples/             # two example images per imaging protocol
    ├── wheat/
    ├── soy_protocol_1/
    ├── soy_protocol_2/
    └── manifest.csv
```

## Model

| Item | Value |
|---|---|
| Architecture | RF-DETR-Seg Small (DINOv2 ViT-S backbone, 33.4 M parameters) |
| Training data | soybean (protocols 1 and 2) + wheat, single class `trichome` |
| Input resolution | 1272 × 1272 (images are resized internally) |
| Confidence threshold | 0.35 (selected on the validation set) |
| Checkpoint | `weights/rfdetrseg_small_wheat_soy.pth` |

On the held-out test sets reported in the article this model reached a counting F1 of 0.967 and a mean absolute error (MAE) of 2.26 trichomes per image on soybean (42 images), and a counting F1 of 0.902 and an MAE of 4.38 on wheat (29 images). Predicted trichome lengths correlate with lengths measured on manual annotation (Pearson r = 0.89–0.97), but on soybean images they are systematically underestimated (regression slope 0.82–0.84), mainly because thin trichome tips are segmented incompletely. See the article for details.

## Training data

The annotated images (polygon annotation of individual trichomes) are available on the Roboflow platform in the project **lhdetect3**, dataset **soy_wheat**: <https://app.roboflow.com/lhdetect3>.

The three imaging protocols are:

| Folder in `examples/` | Crop | Image size, px | Description |
|---|---|---|---|
| `wheat` | wheat | 1388 × 1040 | leaf fold on a light background, transmitted light |
| `soy_protocol_1` | soybean | 1920 × 1370 or 1370 × 1920 (also 2× larger) | dark background, ruler, white panel with acquisition metadata |
| `soy_protocol_2` | soybean | 3584 × 2748 | dark background, ruler, no metadata panel |

## Installation

Requirements: Python 3.11, PyTorch 2.8 and an NVIDIA GPU with CUDA (recommended; inference on CPU works but is slow).

1. Install [Git LFS](https://git-lfs.com) — it is required to download the model weights:

   ```bash
   git lfs install
   ```

2. Clone the repository (the weights are downloaded automatically by Git LFS):

   ```bash
   git clone https://github.com/genaev/lhdetect3.git
   cd lhdetect3
   git lfs pull   # only needed if the weights were not downloaded during cloning
   ```

   Check that `weights/rfdetrseg_small_wheat_soy.pth` is about 129 MB; a file of a few hundred bytes is a Git LFS pointer, which means the weights have not been downloaded.

3. Create an environment and install the dependencies:

   ```bash
   conda create -n lhdetect3 python=3.11
   conda activate lhdetect3
   pip install -r requirements.txt
   ```

   If needed, install a PyTorch build matching your CUDA version first, following <https://pytorch.org/get-started/locally/>.

## Usage

### Quick start

Run the model on the example images:

```bash
python infer.py
```

The script processes all images in `examples/` and writes `predictions.csv` to the repository folder.

### Your own images

```bash
python infer.py /path/to/images --output results.csv
```

The input can be a single image or a folder; folders are searched recursively for `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff` and `.bmp` files. If the output file already exists, images listed in it are skipped, so an interrupted run can be resumed by repeating the same command.

### Command-line options

| Option | Default | Meaning |
|---|---|---|
| `input` | `examples/` | image file or folder |
| `--output` | `predictions.csv` | output CSV file |
| `--weights` | `weights/rfdetrseg_small_wheat_soy.pth` | model checkpoint |
| `--conf` | 0.35 | confidence threshold for detections |
| `--gpu` | 0 | CUDA device index; use `-1` to run on CPU |
| `--min-area` | 20 | minimal mask area, px; smaller masks are discarded |
| `--contour-dilate-px` | 3 | isotropic dilation of a mask before skeletonization, px |
| `--dilate-px` | 5 | dilation along the principal axis of a mask (extends tips), px |
| `--smooth-sigma` | 6.0 | Gaussian smoothing of the centerline, px |
| `--trim-angle` | 60 | sharp bends at the centerline ends above this angle are trimmed, degrees |
| `--merge-dist-px` | 60 | maximal gap between collinear fragments that are merged into one trichome, px |
| `--merge-max-angle` | 30 | maximal angle between merged fragments, degrees |
| `--merge-max-turn` | 40 | maximal direction change at the junction of merged fragments, degrees |

The post-processing parameters were used with their default values in the article.

## Output

The output is a CSV file with one row per image:

```text
image,count,lengths_px
wheat/1520_3_1_1a.png,36,84.2;91.0;...
```

| Column | Description |
|---|---|
| `image` | image path relative to the input folder |
| `count` | number of trichomes detected in the image |
| `lengths_px` | centerline length of each trichome in pixels, separated by semicolons |

Lengths are given in pixels of the original image. To convert them to micrometres, multiply by the image scale (µm per pixel) of your imaging setup.

## Method summary

For every image the model predicts instance masks of trichomes with a confidence of at least 0.35. Each mask is then processed as follows: holes are filled; the mask is dilated isotropically and along its principal axis, which restores thin tips; the longest path of the mask skeleton is taken as the centerline; the path is cleaned of U-turns and kinks, simplified, smoothed and trimmed at sharp ends. Fragments whose centerlines continue each other are merged into one trichome, and overlapping duplicate masks are removed. The number of remaining instances is reported as `count`, and their centerline lengths as `lengths_px`.

## Limitations

- The model was trained on leaf-fold images of soybean and wheat obtained with the three protocols listed above. For other species or imaging setups, the accuracy should be checked on a manually annotated sample.
- Trichome lengths on soybean images are underestimated by about 20% on average; for absolute measurements we recommend calibrating the lengths with a linear regression on a small annotated sample.
- Lengths are reported in pixels; calibration to physical units is up to the user.

## Citation

If you use this code or model, please cite the article (full reference will be added after publication):

```bibtex
@article{genaev2026lhdetect3,
  author  = {Genaev, Mikhail A. and Afonnikov, Dmitry A.},
  title   = {Quantitative assessment of soybean leaf pubescence from {RGB} images using deep learning and instance segmentation},
  journal = {Plants},
  year    = {2026},
  note    = {Submitted}
}
```

## License

The model is built with [RF-DETR](https://github.com/roboflow/rf-detr) (Apache License 2.0). The license of this repository will be specified in the `LICENSE` file.

## Contact

Mikhail A. Genaev, Institute of Cytology and Genetics, Siberian Branch of the Russian Academy of Sciences, Novosibirsk, Russia — mag@bionet.nsc.ru
