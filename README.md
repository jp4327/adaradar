
<div align="center">

# AdaRadar: Adaptive Radar Data Compression (CVPR 2026)

[Jinho Park](https://scholar.google.com/citations?user=vkWRJIAAAAAJ&hl=en)$^{1}$ &nbsp;&nbsp;&nbsp; [Se Young Chun](https://scholar.google.com/citations?user=3jLuG64AAAAJ&hl=en)$^{2}$ &nbsp;&nbsp;&nbsp; [Mingoo Seok](https://scholar.google.com/citations?user=OPECx0sAAAAJ&hl=en)$^{1}$

$^{1}$**Columbia University** &nbsp;&nbsp;&nbsp;&nbsp; $^{2}$**Seoul National University**

</div>

[![project_page](https://img.shields.io/badge/-project%20page-skyblue)](https://jp4327.github.io/adaradar/)

This repository contains the evaluation code for the following paper: AdaRadar: Rate Adaptive Spectral Compression for Radar-based Perception

<img src="docs/static/images/abstract.png" alt="overview" width="700"/>

>Radar is a critical perception modality in autonomous driving systems due to its all-weather characteristics and ability to measure range and Doppler velocity. However, the sheer volume of high-dimensional raw radar data saturates the communication link to the computing engine (e.g., an NPU), which is often a low-bandwidth interface with data rate provisioned only for a few low-resolution range-Doppler frames. A generalized codec for utilizing high-dimensional radar data is notably absent, while existing image-domain approaches are unsuitable, as they typically operate at fixed compression ratios and fail to adapt to varying or adversarial conditions. In light of this, we propose radar data compression with adaptive feedback. It dynamically adjusts the compression ratio by performing gradient descent from the proxy gradient of detection confidence with respect to the compression rate. We employ a zeroth-order gradient approximation as it enables gradient computation even with non-differentiable core operations--pruning and quantization. This also avoids transmitting the gradient tensors over the band-limited link, which, if estimated, would be as large as the original radar data. In addition, we have found that radar feature maps are heavily concentrated on a few frequency components. Thus, we apply the discrete cosine transform to the radar data cubes and selectively prune out the coefficients effectively. We preserve the dynamic range of each radar patch through scaled quantization. Combining those techniques, our proposed online adaptive compression scheme achieves over 100x feature size reduction at minimal performance drop (~1%p). We validate our results on the RADIal, CARRADA, and Radatron datasets.


## Installation

      $git clone https://github.com/jp4327/adaradar
      $conda env create --file environment.yaml
      $conda activate adaradar

<!-- ## Data Preparation

The code in this repository is tested on [KITTI](https://www.cvlibs.net/datasets/kitti/) Odometry dataset. The IMU data after pre-processing is provided under `data/imus`. To download the images and poses, please run

      $cd data
      $source data_prep.sh 

After running the script,`data` folder shall look like the following:

```
data
├── data_prep.sh
├── imus
│   ├── 00.mat
│   ├── 01.mat
│   ...
│   └── 10.mat
├── poses
│   ├── 00.txt
│   ├── 01.txt
│   ...
│   └── 10.txt
└── sequences
    ├── 00
    │   ├── calib.txt
    │   ├── image_2
    │   │   ├── 000000.png
    │   │   ├── 000001.png
    |   |   | ...
    │   ├── image_3
    │   └── times.txt
    ├── 01
    ...
    └── 10
```

## Evaluation

Run pre-trained model with the following command:

      $python test.py

The resulting CLI output looks like:

```
Seq: 05, t_rel: 4.2233, r_rel: 1.3760, t_rmse: 0.0564, r_rmse: 0.0744, 
Seq: 07, t_rel: 3.2585, r_rel: 2.6176, t_rmse: 0.0594, r_rmse: 0.0873, 
Seq: 10, t_rel: 3.0715, r_rel: 1.2113, t_rmse: 0.0611, r_rmse: 0.0859, 
``` -->


## Reference

If you find our work useful in your research, please consider citing our paper:

> Jinho Park, Se Young Chun, Mingoo Seok, "AdaRadar: Rate Adaptive Spectral Compression for Radar-based Perception", CVPR 2026

    @inproceedings{park2026adaradar,
        title={AdaRadar: Rate Adaptive Spectral Compression for Radar-based Perception},
        author={Park, Jinho and Chun, Se Young and Seok, Mingoo},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        year={2026},
    }