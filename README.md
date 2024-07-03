## ___***MVOC: a training-free multiple video object composition method with diffusion models***___

[comment]: <> (![]&#40;asset/logo.png&#41;)

<img src="assets/logo.png" style="height:100px" />

[![arXiv](https://img.shields.io/badge/arXiv-2406.15829-b31b1b.svg)](https://arxiv.org/abs/2406.15829)  [![License](https://img.shields.io/badge/License-MIT-yellow)]() 

[**🌐 Homepage**](https://sobeymil.github.io/mvoc.com/)  |  [**📖 arXiv**](https://arxiv.org/abs/2406.15829)

This repo is an official pytorch implementation of the paper "[MVOC: a training-free multiple video object composition method with diffusion models](https://arxiv.org/abs/2406.15829)"

## Introduction

MVOC is a training-free multiple video object composition framework aimed at achieving visually harmonious and temporally consistent results.

<img src="https://sobeymil.github.io/mvoc.com/mvoc_intro.png" />

Given multiple video objects (e.g. Background, Object1, Object2), our method enables presenting the interaction effects between multiple video objects and maintaining the motion and identity consistency of each object in the composited video.

## ▶️ Quick Start for MVOC

### Environment

Clone this repo and prepare Conda environment using the following commands:

```bash
git clone https://github.com/SobeyMIL/MVOC

cd MVOC
conda env create -f environment.yml
```

### Pretrained model

We use i2vgen-xl to inverse the videos and compose them in a training-free manner. Download it from [huggingface](https://huggingface.co/ali-vilab/i2vgen-xl/tree/main) and put it at `i2vgen-xl/checkpoints`.

### Video Composition

We offer the videos we use in the paper, you can find it at `demo`

First, you need to get the latent representation of the source videos, we offer the inversion config file at `i2vgen-xl/configs/group_inversion/group_config.json`.

Then you can run the following command:

```bash
cd i2vgen-xl/scripts
bash run_group_ddim_inversion.sh
bash run_group_composition.sh
```

## Results

We provide some composition results in this repo as below.

| Demo        | Collage                             | Our result                      |
| ----------- | ----------------------------------- | ------------------------------- |
| boat_surf   | ![](assets/boat_surf_collage.gif)   | ![](demo/boat_surf/video.gif)   |
| crane_seal  | ![](assets/crane_seal_collage.gif)  | ![](demo/crane_seal/video.gif)  |
| duck_crane  | ![](assets/duck_crane_collage.gif)  | ![](demo/duck_crane/video.gif)  |
| monkey_swan | ![](assets/monkey_swan_collage.gif) | ![](demo/monkey_swan/video.gif) |
| rider_deer  | ![](assets/rider_deer_collage.gif)  | ![](demo/rider_deer/video.gif)  |
| robot_cat   | ![](assets/robot_cat_collage.gif)   | ![](demo/robot_cat/video.gif)   |
| seal_bird   | ![](assets/seal_bird_collage.gif)   | ![](demo/seal_bird/video.gif)   |

## 🖊️ Citation

Please kindly cite our paper if you use our code, data, models or results:

```bibtex
@inproceedings{wang2024mvoc,
        title = {MVOC: a training-free multiple video object composition method with diffusion models},
        author = {Wei Wang and Yaosen Chen and Yuegen Liu and  Qi Yuan and  Shubin Yang and  Yanru Zhang},
        year = {2024},
        booktitle = {arxiv}
}
```

## 🎫 License
This project is released under [the MIT License](LICENSE).
## 💞 Acknowledgements

The code is built upon the below repositories, we thank all the contributors for open-sourcing.

* [diffusers](https://github.com/huggingface/diffusers)
* [AnyV2V](https://github.com/TIGER-AI-Lab/AnyV2V)
* [i2vgen-xl](https://github.com/ali-vilab/VGen)