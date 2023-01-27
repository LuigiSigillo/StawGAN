# StawGAN: Structural-Aware Generative Adversarial Networks for Infrared Image Translation
Official PyTorch repository for StawGAN: Structural-Aware Generative Adversarial Networks for Infrared Image Translation, ISCAS 2023

[[IEEEXplore]()][[ArXiv preprint]()]

Luigi Sigillo, Eleonora Grassucci, and Danilo Comminiello

[![PWC]()]()

### Abstract :bookmark_tabs:

This paper addresses the problem of translating night-time thermal infrared images, which are the most adopted image modalities to analyze night-time scenes, to daytime color images (NTIT2DC), which provide better perceptions of objects.
We introduce a novel model that focuses on enhancing the quality of the target generation without merely colorizing it. The proposed structural aware (StawGAN) enables the translation of better-shaped and high-definition objects in the target domain.
We test our model on aerial images of the DroneVeichle dataset containing RGB-IR paired images.
The proposed approach produces a more accurate translation with respect to other state-of-the-art image translation models.

### Model Architecture :clapper:
![Architecture](StawGAN_arch.png)

### Results :bar_chart:
| Model                                    | FID :arrow_down:      | IS :arrow_up:         | PSNR :arrow_up:     | SSIM :arrow_up:    |
|------------------------------------------|----------------------|----------------------|-----------------------|----------------------|
| pix2pixHD| $\mathbf{0.0259}$    | $4.2223$             | $11.2101$             | $0.2125$             |
| StarGAN v2       | $0.4476	$            | $2.7190$             | $11.2211$             | $\underline{0.2297}$ |
| PearlGAN                   | $0.0743$             | $3.9441$             | $10.8925$             | $0.2046$             |
| TarGAN            | $0.1177$             | $\underline{4.3441}$ | $\underline{11.7085}$ | $0.2382$             |
| StawGAN                                  | $\underline{0.0111}$ | $\mathbf{4.4445}$    | $\mathbf{11.8251}$    | $\mathbf{0.2453}$    |

### How to run experiments :computer:

First, please install the requirements:

```pip install -r requirements.txt```



### Cite

Please cite our work if you found it useful:

```
@INPROCEEDINGS{9892119,}
```


