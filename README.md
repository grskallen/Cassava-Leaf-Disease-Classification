# Cassava Leaf Disease Classification  
This repository hosts the codebase for the "Cassava Leaf Disease Classification"  
challenge hosted on Kaggle.  

## News
+ 2024/6/11 This is the first upload. This version is trained using only ResNet50, with approximately 8GB of GPU memory usage and an accuracy rate of over 65%. Subsequent updates will include code for data augmentation.

## Disclaimer  
Please note that this code is not an official  
release from Kaggle or any other authoritative entity.  
  
As the uploader is a novice, the code may contain imperfections  
or lack refinement.  
  
I warmly invite all to contribute enhancements to this codebase and  
if anyone encounters any issues, feel free to raise them here as an issue.  
  
## Implementation method
**a. Create a conda virtual environment and activate it.**
```shell
conda create -n your-env python=3.8 -y
conda activate your-env
```
**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
**c. Install pandas, numpy, pillow**
```
pip install numpy==1.26.4 pandas==2.2.2 pillow==10.3.0
```
**d. Train your model**
```
python run_by_resnet50.py
```
