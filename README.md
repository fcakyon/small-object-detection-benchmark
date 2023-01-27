# small-object-detection-benchmark

<a href="https://ieeexplore.ieee.org/document/9897990"><img src="https://img.shields.io/badge/DOI-10.1109%2FICIP46576.2022.9897990-orange.svg" alt="ci">
<a href="https://twitter.com/fcakyon"><img src="https://img.shields.io/badge/twitter-fcakyon_-blue?logo=twitter&style=flat" alt="fcakyon twitter"></a>

🔥 our paper has been presented in ICIP 2022 Bordeaux, France (16-19 October 2022)

## summary

small-object-detection benchmark on visdrone and xview datasets using [fcos](https://arxiv.org/abs/1904.01355), [vfnet](https://arxiv.org/abs/1810.05943) and [tood](https://arxiv.org/abs/2108.07755) detectors

refer to [Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection](https://ieeexplore.ieee.org/document/9897990) for full technical analysis

## citation

If you use any file/result from this repo in your work, please cite it as:

```
@article{akyon2022sahi,
  title={Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection},
  author={Akyon, Fatih Cagatay and Altinuc, Sinan Onur and Temizel, Alptekin},
  journal={2022 IEEE International Conference on Image Processing (ICIP)},
  doi={10.1109/ICIP46576.2022.9897990},
  pages={966-970},
  year={2022}
}
```

## visdrone results

refer to table 1 in [Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection](https://ieeexplore.ieee.org/document/9897990) for more detail on visdrone results

[fcos_fi_visdrone_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/fcos_fi_visdrone_results.zip
[fcos_sahi_po_visdrone_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/fcos_sahi_po_visdrone_results.zip
[fcos_sahi_fi_po_visdrone_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/fcos_sahi_fi_po_visdrone_results.zip
[fcos_sf_sahi_po_visdrone_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/fcos_sf_sahi_po_visdrone_results.zip
[fcos_sf_sahi_fi_po_visdrone_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/fcos_sf_sahi_fi_po_visdrone_results.zip

[vfnet_fi_visdrone_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/vfnet_fi_visdrone_results.zip
[vfnet_sahi_po_visdrone_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/vfnet_sahi_po_visdrone_results.zip
[vfnet_sahi_fi_po_visdrone_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/vfnet_sahi_fi_po_visdrone_results.zip
[vfnet_sf_sahi_po_visdrone_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/vfnet_sf_sahi_po_visdrone_results.zip
[vfnet_sf_sahi_fi_po_visdrone_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/vfnet_sf_sahi_fi_po_visdrone_results.zip

[tood_fi_visdrone_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/tood_fi_visdrone_results.zip
[tood_sahi_visdrone_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/tood_sahi_visdrone_results.zip
[tood_sahi_po_visdrone_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/tood_sahi_po_visdrone_results.zip
[tood_sahi_fi_visdrone_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/tood_sahi_fi_visdrone_results.zip
[tood_sahi_fi_po_visdrone_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/tood_sahi_fi_po_visdrone_results.zip

[tood_sf_fi_visdrone_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/tood_sf_fi_visdrone_results.zip
[tood_sf_sahi_visdrone_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/tood_sf_sahi_visdrone_results.zip
[tood_sf_sahi_po_visdrone_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/tood_sf_sahi_po_visdrone_results.zip
[tood_sf_sahi_fi_visdrone_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/tood_sf_sahi_fi_visdrone_results.zip
[tood_sf_sahi_fi_po_visdrone_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/tood_sf_sahi_fi_po_visdrone_results.zip

[tood_sf_visdrone_checkpoint_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.2/tood_sf_visdrone.pth
[fcos_sf_visdrone_checkpoint_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.2/fcos_sf_visdrone.pth

[my_twitter_url]: https://twitter.com/fcakyon

|setup |AP<sub>50</sub> |AP<sub>50</sub>s |AP<sub>50</sub>m |AP<sub>50</sub>l | results | checkpoints |
|--- |--- |--- |--- |--- |--- |--- |
|FCOS+FI |25.8 |14.2 |39.6 |45.1 | [download][fcos_fi_visdrone_results_url] | [request][my_twitter_url] |
|FCOS+SAHI+PO |29.0 |18.9 |41.5 |46.4 | [download][fcos_sahi_po_visdrone_results_url] | [request][my_twitter_url] |
|FCOS+SAHI+FI+PO |31.0 |19.8 |44.6 |49.0 | [download][fcos_sahi_fi_po_visdrone_results_url] | [request][my_twitter_url] |
|FCOS+SF+SAHI+PO |38.1 |25.7 |54.8 |56.9 | [download][fcos_sf_sahi_po_visdrone_results_url] | [download][fcos_sf_visdrone_checkpoint_url] |
|FCOS+SF+SAHI+FI+PO |38.5 |25.9 |55.4 |59.8 | [download][fcos_sf_sahi_fi_po_visdrone_results_url] | [download][fcos_sf_visdrone_checkpoint_url] |
|--- |--- |--- |--- |--- |--- |--- |
|VFNet+FI |28.8 |16.8 |44.0 |47.5 | [download][vfnet_fi_visdrone_results_url] | [request][my_twitter_url] |
|VFNet+SAHI+PO |32.0 |21.4 |45.8 |45.5 | [download][vfnet_sahi_po_visdrone_results_url] | [request][my_twitter_url] |
|VFNet+SAHI+FI+PO |33.9 |22.4 |49.1 |49.4 | [download][vfnet_sahi_fi_po_visdrone_results_url] | [request][my_twitter_url] |
|VFNet+SF+SAHI+PO |41.9 |29.7 |58.8 |60.6 | [download][vfnet_sf_sahi_po_visdrone_results_url] | [request][my_twitter_url] |
|VFNet+SF+SAHI+FI+PO |42.2 |29.6 |59.2 |63.3 | [download][vfnet_sf_sahi_fi_po_visdrone_results_url] | [request][my_twitter_url] |
|--- |--- |--- |--- |--- |--- |--- |
|TOOD+FI |29.4 |18.1 |44.1 |50.0 | [download][tood_fi_visdrone_results_url] | [request][my_twitter_url] |
|TOOD+SAHI |31.9 |22.6 |44.0 |45.2 | [download][tood_sahi_visdrone_results_url] | [request][my_twitter_url] |
|TOOD+SAHI+PO |32.5 |22.8 |45.2 |43.6 | [download][tood_sahi_po_visdrone_results_url] | [request][my_twitter_url] |
|TOOD+SAHI+FI |34.6 |23.8 |48.5 |53.1 | [download][tood_sahi_fi_visdrone_results_url] | [request][my_twitter_url] |
|TOOD+SAHI+FI+PO |34.7 |23.8 |48.9 |50.3| [download][tood_sahi_fi_po_visdrone_results_url] | [request][my_twitter_url] |
|TOOD+SF+FI |36.8 |24.4 |53.8 |66.4 | [download][tood_sf_fi_visdrone_results_url] | [download][tood_sf_visdrone_checkpoint_url] |
|TOOD+SF+SAHI |42.5 |31.6 |58.0 |61.1 | [download][tood_sf_sahi_visdrone_results_url] | [download][tood_sf_visdrone_checkpoint_url] |
|TOOD+SF+SAHI+PO |43.1 |31.7 |59.0 |60.2 | [download][tood_sf_sahi_po_visdrone_results_url] | [download][tood_sf_visdrone_checkpoint_url] |
|TOOD+SF+SAHI+FI |43.4 |31.7 |59.6 |65.6 | [download][tood_sf_sahi_fi_visdrone_results_url] | [download][tood_sf_visdrone_checkpoint_url] |
|TOOD+SF+SAHI+FI+PO |43.5 |31.7 |59.8 |65.4 | [download][tood_sf_sahi_fi_po_visdrone_results_url] | [download][tood_sf_visdrone_checkpoint_url] |

## xview results

refer to table 2 in [Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection](https://ieeexplore.ieee.org/document/9897990) for more detail on xview results

[fcos_fi_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/fcos_fi_xview_results.zip
[fcos_sf_sahi_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/fcos_sf_sahi_xview_results.zip
[fcos_sf_sahi_fi_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/fcos_sf_sahi_fi_xview_results.zip
[fcos_sf_sahi_fi_po_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/fcos_sf_sahi_fi_op_xview_results.zip
[fcos_sf_sahi_po_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/fcos_sf_sahi_op_xview_results.zip

[vfnet_fi_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/vfnet_fi_xview_results.zip
[vfnet_sf_sahi_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/vfnet_sf_sahi_xview_results.zip
[vfnet_sf_sahi_fi_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/vfnet_sf_sahi_fi_xview_results.zip
[vfnet_sf_sahi_fi_po_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/vfnet_sf_sahi_fi_op_xview_results.zip
[vfnet_sf_sahi_po_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/vfnet_sf_sahi_op_xview_results.zip

[tood_fi_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/tood_fi_xview_results.zip
[tood_sf_sahi_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/tood_sf_sahi_xview_results.zip
[tood_sf_sahi_fi_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/tood_sf_sahi_fi_xview_results.zip
[tood_sf_sahi_fi_po_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/tood_sf_sahi_fi_op_xview_results.zip
[tood_sf_sahi_po_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/tood_sf_sahi_op_xview_results.zip

[fcos_sf_xview_checkpoint_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.2/fcos_sf_xview.pth
[vfnet_sf_xview_checkpoint_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.2/vfnet_sf_xview.pth
[tood_sf_xview_checkpoint_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.2/tood_sf_xview.pth

|setup |AP<sub>50</sub> |AP<sub>50</sub>s |AP<sub>50</sub>m |AP<sub>50</sub>l | results | checkpoints |
|--- |--- |--- |--- |--- |--- |--- |
|FCOS+FI |2.20 |0.10 |1.80 |7.30 | [download][fcos_fi_xview_results_url] | [request][my_twitter_url] |
|FCOS+SF+SAHI |15.8 |11.9 |18.4 |11.0 | [download][fcos_sf_sahi_xview_results_url] | [download][fcos_sf_xview_checkpoint_url] |
|FCOS+SF+SAHI+PO |17.1 |12.2 |20.2 |12.8 | [download][fcos_sf_sahi_po_xview_results_url] | [download][fcos_sf_xview_checkpoint_url] |
|FCOS+SF+SAHI+FI |15.7 |11.9 |18.4 |14.3 | [download][fcos_sf_sahi_fi_xview_results_url] | [download][fcos_sf_xview_checkpoint_url] |
|FCOS+SF+SAHI+FI+PO |17.0 |12.2 |20.2 |15.8 | [download][fcos_sf_sahi_fi_po_xview_results_url] | [download][fcos_sf_xview_checkpoint_url] |
|--- |--- |--- |--- |--- |--- |--- |
|VFNet+FI |2.10 |0.50 |1.80 |6.80 | [download][vfnet_fi_xview_results_url] | [request][my_twitter_url] |
|VFNet+SF+SAHI | 16.0 |11.9 |17.6 |13.1 | [download][vfnet_sf_sahi_xview_results_url] | [download][vfnet_sf_xview_checkpoint_url] |
|VFNet+SF+SAHI+PO |17.7| 13.7 |19.7 |15.4 | [download][vfnet_sf_sahi_po_xview_results_url] | [download][vfnet_sf_xview_checkpoint_url] |
|VFNet+SF+SAHI+FI |15.8 |11.9 |17.5 |15.2 | [download][vfnet_sf_sahi_fi_xview_results_url] | [download][vfnet_sf_xview_checkpoint_url] |
|VFNet+SF+SAHI+FI+PO |17.5 |13.7 |19.6 |17.6 | [download][vfnet_sf_sahi_fi_po_xview_results_url] | [download][vfnet_sf_xview_checkpoint_url] |
|--- |--- |--- |--- |--- |--- |--- |
|TOOD+FI |2.10 |0.10 |2.00 |5.20 | [download][tood_fi_xview_results_url] | [request][my_twitter_url] |
|TOOD+SF+SAHI |19.4 |14.6 |22.5 |14.2 | [download][tood_sf_sahi_xview_results_url] | [download][tood_sf_xview_checkpoint_url] |
|TOOD+SF+SAHI+PO |20.6 |14.9 |23.6 |17.0 | [download][tood_sf_sahi_po_xview_results_url] | [download][tood_sf_xview_checkpoint_url] |
|TOOD+SF+SAHI+FI |19.2 |14.6 |22.3 |14.7 | [download][tood_sf_sahi_fi_xview_results_url] | [download][tood_sf_xview_checkpoint_url] |
|TOOD+SF+SAHI+FI+PO |20.4 |14.9 |23.5 |17.6 | [download][tood_sf_sahi_fi_po_xview_results_url] | [download][tood_sf_xview_checkpoint_url] |

## env setup

install pytorch:

```bash
conda install pytorch=1.10.0 torchvision=0.11.1 cudatoolkit=11.3 -c pytorch
```

install other requirements:

```bash
pip install -r requirements.txt
```

## evaluation

- download desired checkpoint from the urls in readme.

- download xivew or visdrone dataset and convert to COCO format.

- set `MODEL_PATH`, `MODEL_CONFIG_PATH`, `EVAL_IMAGES_FOLDER_DIR`, `EVAL_DATASET_JSON_PATH`, `INFERENCE_SETTING` in [predict_evaluate_analyse script](eval_tools/predict_evaluate_analyse.py) then run the script.

## roadmap

- [x] add train test split support for xview to coco converter
- [x] add mmdet config files (fcos, vfnet and tood) for xview training (9 train experiments)
- [x] add mmdet config files (fcos, vfnet and tood) for visdrone training (9 train experiments)
- [x] add coco result.json files, classwise coco eval results error analysis plots for all xview experiments
- [x] add coco result.json files, classwise coco eval results error analysis plots for all visdrone experiments
- [X] add .py scripts for inference + evaluation + error analysis using `sahi`
