# Pneumothorax Segmentation in Routine Computed Tomography Based on Deep Neural Networks (ICoIAS'2021)
This library contains the original implementation of **'Pneumothorax Segmentation in Routine Computed Tomography Based on Deep Neural Networks'** in Keras (Tensorflow as backend).

paper link: [https://ieeexplore.ieee.org/abstract/document/9527604](https://ieeexplore.ieee.org/abstract/document/9527604)
## Required libraries
- Python == 3.6
- Keras == 2.4.3
- tensorflow-gpu == 2.2.0rc3
- h5py == 2.10.0
- numpy == 1.19.4
- pydensecrf == 1.0rc3
- SimpleITK == 2.0.1
- opencv-python == 4.4.0.46
## Implemented Models
- U-Net
- dilated U-Net
- ResNet34_Unet
- ResNet50_Unet
- PSPNet
- Attention U-Net
- UNet++
- MultiResUNet
- MFP-Unet
- UNet3+
## Segmentation Results
![image](https://github.com/FreedomXL/Pneumothorax-Segmentation-Deep-Learning/blob/master/images/Dice%20Coefficient.png)

![image](https://github.com/FreedomXL/Pneumothorax-Segmentation-Deep-Learning/blob/master/images/Visualization%20Results.png)
<p align='center'>Figure 1: Visualization of Pneumothorax Segmentation Results</p>

## Citation
```
@inproceedings{wu2021pneumothorax,
  title={Pneumothorax Segmentation in Routine Computed Tomography Based on Deep Neural Networks},
  author={Wu, Wenbin and Liu, Guanjun and Liang, Kaiyi and Zhou, Hui},
  booktitle={2021 4th International Conference on Intelligent Autonomous Systems (ICoIAS)},
  pages={78--83},
  year={2021},
  organization={IEEE}
}
```
