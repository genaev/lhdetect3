# Trichome counting with RF-DETR-Seg

Inference package for the best counting model: **RF-DETR-Seg Small**, fine-tuned on labeled soybean and wheat images (`trichomes` class). The checkpoint is `weights/rfdetrseg_small_wheat_soy.pth` (resolution 1272, confidence threshold 0.35).

`infer.py` writes one CSV row per image:

```text
image,count,lengths_px
wheat/1520_3_1_1a_png.rf.yqETPbIIqlfoXhtekWTv.png,36,84.2;91.0
```

`count` is the number of trichomes. `lengths_px` lists their centerline lengths in pixels, separated by semicolons. Lengths come from the skeleton of each predicted mask after hole filling, tip dilation, smoothing, and merging of collinear fragments.

## Examples

`examples/` contains two images from each acquisition protocol. See `examples/manifest.csv` for size and source path.

- `examples/wheat/` — 1388 x 1040
- `examples/soy_protocol_1/` — 1920 x 1370, white technical band
- `examples/soy_protocol_2/` — 3584 x 2748, no technical band

## Run

Use the `rfdetr_env` environment (Python 3.11, PyTorch with CUDA, `rfdetr==1.5.2`):

```bash
conda activate rfdetr_env
python infer.py
```

This reads `examples/` and writes `predictions.csv`. For another folder:

```bash
python infer.py /path/to/images --output results.csv --gpu 0
```

The script resumes an existing CSV and skips images already listed in it.

## GitHub

The checkpoint is 129 MB. GitHub rejects files larger than 100 MB, so install [Git LFS](https://git-lfs.com) before the first commit that adds the weights:

```bash
git lfs install
git add .gitattributes
git add weights/rfdetrseg_small_wheat_soy.pth
```

RF-DETR is Apache-2.0. These research weights are not the public Roboflow release.
