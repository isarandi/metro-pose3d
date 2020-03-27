# MeTRo 3D Human Pose Estimator

### What is this?

Code to train and evaluate the MeTRo method, proposed in our paper
"Metric-Scale Truncation-Robust Heatmaps for 3D Human Pose Estimation" (Sárándi et al., 2020).
 A preprint of the paper is on arXiv: https://arxiv.org/abs/2003.02953

### What does it do?

It takes a single **RGB image of a person as input** and returns the **3D coordinates of 17 body joints** relative to the pelvis. The coordinates are estimated in millimeters directly. Also, it always returns a complete pose by guessing joint positions even outside of the image boundaries (truncation).  
 
### How do I run it?
There is a small, self-contained script `inference.py` with minimal dependencies (just TensorFlow + NumPy), which can perform inference with a pretrained, exported model. Use it as follows:

```bash
wget https://omnomnom.vision.rwth-aachen.de/data/metro-pose3d/coco_mpii_h36m_3dhp_cmu_3dpw_resnet50_upperbodyaug_stride16.pb
./inference.py --model-path=coco_mpii_h36m_3dhp_cmu_3dpw_resnet50_upperbodyaug_stride16.pb
```

### How do I train it?
See [DEPENDENCIES.md](docs/DEPENDENCIES.md) for installing the dependencies. Then follow [DATASETS.md](docs/DATASETS.md) to download and prepare the training and test data.
Finally, see [TRAINING.md](docs/TRAINING.md) for instructions on running experiments.

### How do I cite it?
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
