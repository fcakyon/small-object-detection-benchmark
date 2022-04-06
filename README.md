# sahi-benchmark

sahi benchmark on visdrone and xview datasets using [fcos](https://arxiv.org/abs/1904.01355), [vfnet](https://arxiv.org/abs/1810.05943) and [tood](https://arxiv.org/abs/2108.07755) detectors

# xview results

[fcos_fi_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/fcos_fi_xview_results.zip
[fcos_sf_sahi_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/fcos_sf_sahi_xview_results.zip
[fcos_sf_sahi_fi_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/fcos_sf_sahi_fi_xview_results.zip
[fcos_sf_sahi_fi_op_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/fcos_sf_sahi_fi_op_xview_results.zip
[fcos_sf_sahi_op_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/fcos_sf_sahi_op_xview_results.zip

[vfnet_fi_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/vfnet_fi_xview_results.zip
[vfnet_sf_sahi_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/vfnet_sf_sahi_xview_results.zip
[vfnet_sf_sahi_fi_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/vfnet_sf_sahi_fi_xview_results.zip
[vfnet_sf_sahi_fi_op_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/vfnet_sf_sahi_fi_op_xview_results.zip
[vfnet_sf_sahi_op_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/vfnet_sf_sahi_op_xview_results.zip

[tood_fi_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/tood_fi_xview_results.zip
[tood_sf_sahi_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/tood_sf_sahi_xview_results.zip
[tood_sf_sahi_fi_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/tood_sf_sahi_fi_xview_results.zip
[tood_sf_sahi_fi_op_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/tood_sf_sahi_fi_op_xview_results.zip
[tood_sf_sahi_op_xview_results_url]: https://github.com/fcakyon/sahi-benchmark/releases/download/v0.0.1/tood_sf_sahi_op_xview_results.zip

|setup |AP<sub>50</sub> |AP<sub>50</sub>s |AP<sub>50</sub>m |AP<sub>50</sub>l | results |
|--- |--- |--- |--- |--- |--- |
|FCOS+FI |2.20 |0.10 |1.80 |7.30 | [download](fcos_fi_xview_results_url)
|FCOS+SF+SAHI |15.8 |11.9 |18.4 |11.0 | [download](fcos_sf_sahi_xview_results_url)
|FCOS+SF+SAHI+OP |17.1 |12.2 |20.2 |12.8 | [download](fcos_sf_sahi_op_xview_results_url)
|FCOS+SF+SAHI+FI |15.7 |11.9 |18.4 |14.3 | [download](fcos_sf_sahi_fi_xview_results_url)
|FCOS+SF+SAHI+FI+OP |17.0 |12.2 |20.2 |15.8 | [download](fcos_sf_sahi_fi_op_xview_results_url)
|--- |--- |--- |--- |--- |--- |
|VFNet+FI |2.10 |0.50 |1.80 |6.80 | [download](vfnet_fi_xview_results_url)
|VFNet+SF+SAHI |14.8 |11.8 |16.0 |12.5 | [download](vfnet_sf_sahi_xview_results_url)
|VFNet+SF+SAHI+OP |16.5 |10.4 |17.6 |15.7 | [download](vfnet_sf_sahi_op_xview_results_url)
|VFNet+SF+SAHI+FI |14.6 |10.2 |14.5 |15.7 | [download](vfnet_sf_sahi_fi_xview_results_url)
|VFNet+SF+SAHI+FI+OP |15.8 |10.2 |15.8 |17.0 | [download](vfnet_sf_sahi_fi_op_xview_results_url)
|--- |--- |--- |--- |--- |--- |
|TOOD+FI |2.10 |0.10 |2.00 |5.20 | [download](tood_fi_xview_results_url)
|TOOD+SF+SAHI |19.4 |14.6 |22.5 |14.2 | [download](tood_sf_sahi_xview_results_url)
|TOOD+SF+SAHI+OP |20.6 |14.9 |23.6 |17.0 | [download](tood_sf_sahi_op_xview_results_url)
|TOOD+SF+SAHI+FI |19.2 |14.6 |22.3 |14.7 | [download](tood_sf_sahi_fi_xview_results_url)
|TOOD+SF+SAHI+FI+OP |20.4 |14.9 |23.5 |17.6 | [download](tood_sf_sahi_fi_op_xview_results_url)

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
- [x] add coco result.json files, classwise coco eval results error analysis plots for all xview experiments
- [ ] add coco result.json files, classwise coco eval results error analysis plots for all visdrone experiments
- [ ] add .py scripts for inference + evaluation + error analysis using `sahi`
