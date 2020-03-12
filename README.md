## MeTRo 3D Human Pose Estimator

#### What is this?

Code to train and evaluate the MeTRo method, proposed in our paper
"Metric-Scale Truncation-Robust Heatmaps for 3D Human Pose Estimation" (Sárándi et al., 2020).
 A preprint of the paper is on arXiv: https://arxiv.org/abs/2003.02953

#### What does it do?

It's a 3D human pose estimator. It takes a single **RGB image of a person as input** and returns the **3D coordinates of 17 body joints** relative to the pelvis. The coordinates are estimated in millimeters. It always returns a complete pose by guessing joint positions even outside of the image.  
 
#### How do I run it?

Check out the [Live Demo here]().

If you want run inference on your own machine, see [INFERENCE.md](docs/INFERENCE.md).

#### How do I train it?
See [DATASETS.md](docs/DATASETS.md) on how to download and prepare the training and test data.

Then see [TRAINING.md](docs/TRAINING.md) for instructions on running experiments.


#### How do I cite it?
If you use this work, please cite it as:

```bibtex
@inproceedings{Sarandi20FG,
  title={Metric-Scale Truncation-Robust Heatmaps for 3{D} Human Pose Estimation},
  author={S\'ar\'andi, Istv\'an and Linder, Timm and Arras, Kai O. and Leibe, Bastian},
  booktitle={Automatic Face and Gesture Recognition, 2020 IEEE Int. Conf. on},
  year={2020},
  note={in press}
}
```
