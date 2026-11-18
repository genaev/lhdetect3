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

These values refer to the number of instances predicted by the model. Before counting, `infer.py` additionally discards masks smaller than `--min-area` and merges fragments of one trichome, so its counts can differ slightly.

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
image,count,lengths_px
wheat/1520_3_1_1a.png,36,84.2;91.0;...
```

`count` is the number of trichomes; `lengths_px` lists their centerline lengths in pixels, separated by semicolons.
