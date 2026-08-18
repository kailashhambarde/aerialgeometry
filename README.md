# aerialgeometry

**Vision-only camera-geometry prediction from a single aerial image.**

[![arXiv](https://img.shields.io/badge/arXiv-2601.21405-b31b1b.svg)](https://arxiv.org/abs/2601.21405)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

`aerialgeometry` is a command-line tool and a small Python library that takes
one RGB image captured by an aerial platform and predicts the **camera
geometry**:

| Output | Meaning | Values |
| --- | --- | --- |
| `height` | drone **altitude** above ground | meters |
| `distance` | **horizontal distance** to the imaged subject | meters |
| `angle` | **viewing angle** | 30°, 60° or 90° |

It is the self-contained, pip-installable version of the research script
`infer.py` behind

> **Rectifying Geometry-Induced Similarity Distortions for Real-World
> Aerial–Ground Person Re-Identification** — Kailash A. Hambarde,
> Hugo Proença. arXiv:2601.21405.

## About

Aerial–ground person re-identification (AG-ReID) must match people across
drone and ground cameras with extreme differences in altitude, distance and
viewing angle. The GeoReID framework conditions its matching on explicit
*camera geometry*. When a dataset does not provide that metadata, GeoReID
obtains it with a **vision-only multi-task geometry prediction network**
(Fig. 3, Sec. III-B of the paper) — and that is exactly what this package
ships.

A shared **ResNet-50** encoder extracts visual features, followed by two
task-specific heads: a regression head for altitude and horizontal distance,
and a classification head for the viewing angle. This package contains only
the geometry predictor; the full GeoReID framework lives in the
[GeoReID repository](https://github.com/kailashhambarde/GeoReID).

### Model card

| | |
| --- | --- |
| **Task** | Vision-only camera-geometry estimation (no external sensors) |
| **Input** | Single RGB image |
| **Outputs** | altitude, horizontal distance, viewing angle |
| **Backbone** | ResNet-50 (ImageNet-pretrained) |
| **Heads** | 2-output regression + 3-class classification |
| **Preprocessing** | resize 256×256 → center-crop 224×224 → ImageNet normalization |
| **Framework** | PyTorch / torchvision |
| **Held-out metrics** (stored in the released checkpoint) | height MAE **3.39 m** · distance MAE **3.86 m** · angle accuracy **92.3%** |
| **Benchmarks** | AG-ReIDv1, AG-ReIDv2, CARGO, DetReIDX |

> Units are meters because that is how the underlying benchmarks annotate
> geometry. The angle classifier is restricted to the three viewing angles
> used in the datasets: 30°, 60° and 90°.

## Installation

Requires Python 3.9 or later.

```bash
python3 -m pip install aerialgeometry
```

`torch` and `torchvision` are installed as dependencies. Inference runs on
CPU; if a CUDA-capable GPU and a matching PyTorch build are available, it is
used automatically.

To install the latest revision straight from git:

```bash
python3 -m pip install 'git+https://github.com/kailashhambarde/GeoReID'
```

### Model weights

The trained checkpoint (~100 MB) is **not** bundled with the package. The
predictor looks for weights in this order:

1. the `--model-path` CLI argument / `model_path=` Python argument,
2. the `AERIALGEOMETRY_MODEL` environment variable,
3. cached weights at `~/.cache/aerialgeometry/best.pth`,
4. an automatic download from the `AERIALGEOMETRY_MODEL_URL` environment
   variable (e.g. a GitHub release asset or a Hugging Face file).

The simplest setup for end users is to host the checkpoint somewhere public
and export the URL once:

```bash
export AERIALGEOMETRY_MODEL_URL="https://github.com/<you>/GeoReID/releases/download/v1.0/best.pth"
aerialgeometry photos/
# downloads ~/.cache/aerialgeometry/best.pth on first run, then predicts
```

To prepare a slim checkpoint from your training run (drop the optimizer and
scheduler state, keep only what inference needs):

```python
import torch
ckpt = torch.load("runs/drone_predictor_gpu1/best.pth", map_location="cpu", weights_only=False)
torch.save(
    {
        "model_state": ckpt["model_state"],
        "normalization_stats": ckpt["normalization_stats"],
    },
    "best.pth",
)
```

## Quick start

```bash
# every image in a folder (recursively) -> predictions.txt
aerialgeometry photos/

# a single image -> one row on stdout
aerialgeometry photo.jpg
```

```python
from aerialgeometry import Predictor

predictor = Predictor()          # cached / auto-downloaded weights
pred = predictor.predict("photo.jpg")
print(pred.height, pred.distance, pred.angle)   # e.g. 88.41 117.02 60.0
```

## Command line

```
usage: aerialgeometry [-h] [-o OUTPUT] [--model-path MODEL_PATH]
                      [--device DEVICE] [--batch-size BATCH_SIZE]
                      [--version]
                      input

Estimate drone height, distance and viewing angle from aerial images
(arXiv:2601.21405).

positional arguments:
  input                 Folder of images (scanned recursively) or a single
                        image file.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output CSV file. For a folder input the default is
                        <input>/predictions.txt; for a single image the
                        default is stdout.
  --model-path MODEL_PATH
                        Path to the trained checkpoint. Defaults to cached
                        weights; see README for AERIALGEOMETRY_MODEL /
                        AERIALGEOMETRY_MODEL_URL.
  --device DEVICE       Torch device, e.g. 'cpu' or 'cuda:0' (default: auto).
  --batch-size BATCH_SIZE
                        Number of images to process per batch (default: 8).
  --version             Print version number and exit.
```

Examples:

```bash
aerialgeometry photos/ -o out/predictions.csv
aerialgeometry photos/ --model-path best.pth --device cpu
aerialgeometry photos/ --batch-size 32 --device cuda:0     # large jobs on GPU
```

The output is CSV with columns `image_path,height,distance,angle` (identical
to the original `infer.py` `predictions.txt` format, so existing pipelines
keep working):

```
image_path,height,distance,angle
photos/0401_05_04_2024_09_39_90_120_60_...jpg,88.4123,117.0234,60.0
```

## Python API

```python
from aerialgeometry import Predictor

# Uses cached / auto-downloaded weights; pass model_path=... for a checkpoint.
predictor = Predictor(device="cuda:0")   # device is optional, auto-detected

# Single image (file path or PIL image)
pred = predictor.predict("photo.jpg")
print(pred.height)                    # 88.41  (meters)
print(pred.distance)                  # 117.02 (meters)
print(pred.angle)                     # 60.0   (30 / 60 / 90)
print(pred.angle_probabilities)       # [0.01 0.98 0.01] over [30, 60, 90]

# Batch a list of images directly (paths and/or PIL images, order preserved)
preds = predictor.predict_batch(["a.jpg", "b.jpg", some_pil_image])
for pred in preds:
    print(pred.height, pred.distance, pred.angle)

# Scan a folder (processed in batches of 8 by default; adjust for big jobs)
for path, pred in predictor.predict_folder("photos/", batch_size=64):
    print(path, pred.height, pred.distance, pred.angle)
```

The model is also importable for embedding into larger systems:

```python
from aerialgeometry import MultiTaskModel

model = MultiTaskModel()   # ResNet-50 + regression/classification heads
```

## How it works

1. Each image is resized to 256×256 and center-cropped to 224×224, then
   normalized with ImageNet statistics (the same pipeline used in training).
2. The ResNet-50 backbone produces a shared feature vector.
3. The regression head predicts altitude and horizontal distance, denormalized
   with the mean/std statistics stored in the checkpoint.
4. The classification head predicts the viewing angle among {30, 60, 90}°.

The network was trained on aerial imagery where the filenames carry the
ground-truth geometry
(`PID_DD_MM_YYYY_HH_MM_altitude_distance_angle_...`), so it requires no
external sensors at inference time.

> **Numerical note.** On GPUs PyTorch may use TF32 arithmetic by default, so
> GPU predictions can differ from CPU predictions by a few centimeters —
> negligible relative to the model's error, and the angle class is unaffected.

## Results

The full GeoReID framework is evaluated on four aerial–ground person ReID
benchmarks — **AG-ReIDv1**, **AG-ReIDv2**, **CARGO** and **DetReIDX** — where
it improves cross-view matching, especially in high-altitude and large
viewing-angle regimes. Complete tables and ablations are in the
[paper](https://arxiv.org/abs/2601.21405).

## Publishing to PyPI

From the project root:

```bash
python3 -m pip install build twine
python3 -m build                     # builds sdist + wheel into dist/
python3 -m twine upload dist/*       # enter your PyPI credentials
```

Then users can `pip install aerialgeometry`. Remember to host the checkpoint
and set `AERIALGEOMETRY_MODEL_URL` (or document `--model-path`).

## Citation

If you use this tool in your research, please cite:

```bibtex
@article{hambarde2026rectifying,
  title={Rectifying Geometry-Induced Similarity Distortions for Real-World Aerial-Ground Person Re-Identification},
  author={Hambarde, Kailash A and Proen{\c{c}}a, Hugo},
  journal={arXiv preprint arXiv:2601.21405},
  year={2026}
}
```

## Acknowledgements

This work was supported by national funds through FCT – Fundação para a
Ciência e a Tecnologia, I.P., and, when applicable, co-funded by EU funds
under project UID/50008/2025 – Instituto de Telecomunicações.

## License

MIT. See [LICENSE](LICENSE).
