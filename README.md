# LHDetect3

Counting and length measurement of leaf trichomes with an RF-DETR-Seg Small model trained on soybean and wheat leaf-fold images.

Training data: Roboflow project [lhdetect3](https://app.roboflow.com/lhdetect3), dataset `soy_wheat`.

## Installation

```bash
git lfs install
git clone https://github.com/genaev/lhdetect3.git
cd lhdetect3
conda create -n lhdetect3 python=3.11
conda activate lhdetect3
pip install -r requirements.txt
```

The weights `weights/rfdetrseg_small_wheat_soy.pth` (129 MB) are stored with Git LFS. A CUDA GPU is recommended.

## Usage

```bash
python infer.py                                   # example images in examples/
python infer.py /path/to/images --output results.csv
```

Options: `--conf` confidence threshold (default 0.35), `--gpu` CUDA device (`-1` for CPU), `--weights` checkpoint path. Run `python infer.py -h` for all options.

## Output

One CSV row per image:

```text
image,count,lengths_px
wheat/1520_3_1_1a.png,36,84.2;91.0;...
```

`count` is the number of trichomes; `lengths_px` lists their centerline lengths in pixels, separated by semicolons.
