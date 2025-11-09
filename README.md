# Dynamic Privacy-preserving (DyPP) camera
This is the code for the camera model (optics module) in the paper: \
[Learning a Dynamic Privacy-Preserving Camera Robust to Inversion Attacks](https://link.springer.com/chapter/10.1007/978-3-031-72897-6_20). \
Jiacheng Cheng, Xiang Dai, Jia Wan, Nick Antipa, and Nuno Vasconcelos. In ECCV 2024.

If you find this repo useful for your research, please cite this paper as: 
```
@inproceedings{cheng2024learning,
  title={Learning a Dynamic Privacy-Preserving Camera Robust to Inversion Attacks},
  author={Cheng, Jiacheng and Dai, Xiang and Wan, Jia and Antipa, Nick and Vasconcelos, Nuno},
  booktitle={European Conference on Computer Vision},
  pages={349--367},
  year={2024},
  organization={Springer}
}
```

## Dependencies
We implement our model in PyTorch on NVIDIA GPUs. The environment requirements are as bellow:
- PyTorch, version >= 1.11.0
- torchvision, version >= 0.12.0
- timm, version >= 1.0.8


## Acknowledgments
Parts of implementations are based on the following public code: 
- [inversegraphics](https://github.com/polmorenoc/inversegraphics)

