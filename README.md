# sahi-benchmark
sahi benchmark on visdrone and xview datasets using [fcos](https://arxiv.org/abs/1904.01355), [vfnet](https://arxiv.org/abs/1810.05943) and [tood](https://arxiv.org/abs/2108.07755) detectors

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

- [x] add train test split support for xview to coco converter
- [x] add mmdet config files (fcos, vfnet and tood) for xview training (9 train experiments)
- [x] add mmdet config files (fcos, vfnet and tood) for visdrone training (9 train experiments)
- [ ] add .py scripts for inference + evaluation + error analysis using `sahi`