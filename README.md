# sahi-benchmark
sahi benchmark on visdrone and xview datasets using retinanet, yolox and detr detectors

# env setup

install pytorch:

```bash
conda install pytorch=1.10.0 torchvision=0.11.1 cudatoolkit=11.3 -c pytorch
```

install other requirements:

```bash
pip install -r requirements.txt
```

# roadmap

xview:

- [x] add train test split support for xview to coco converter
- [ ] add .sh and .bat scripts for creating sliced xview datasets using `sahi coco slice` command (for slice_size: {300, 400, 500} and overlap_ratio: {0, 0.25})
- [ ] add .sh and .bat scripts for creating sliced visdrone datasets using `sahi coco slice` command (for slice_size: {320, 640, 960} and overlap_ratio: {0, 0.25})
- [ ] add mmdet config files (retinanet, yolox and detr) for xview training (18 experiments)
- [ ] add mmdet config files (retinanet, yolox and detr) for visdrone training (27 experiments)
- [ ] add .py scripts for inference + evaluation + error analysis using `sahi`