# BRI3L: A brightness illusion image dataset for identification and localization of regions of illusory perception

Description: This is a dataset of brightness illusion images, containing : 1) Hermann grid, 2) Simultaneous Brightness Contrast, 3) White’s illusion, 4) Grid illusion, and 5) Induced Grating illusion. There are 22,366 illusion images and corresponding ground-truth localization mask.

<img src="https://github.com/aniket004/BRI3L/blob/main/ill_example_supp.png" width="128" height="128" class="center"/>


# Download dataset:

The dataset is hosted in the link: https://www.cis.jhu.edu/~aroy/Supplementary_BRI3L.zip

Download dataset:
```
!wget https://www.cis.jhu.edu/~aroy/Supplementary_BRI3L.zip
```

unzip dataset:
```
!unzip Supplementary_BRI3L.zip
```

# Tasks:

We perform the task of illusion detection and localization.

# Illusion detection: 

Fot this task, there are 22,366 illusion images and 1149 non-illusion images. Using data augmentation, we increase the non-illusion images to be 3000, and take 500 images from each of the five class of illusions to perform illusion vs non-illusion detection.

# Illusion localization: 

For this task, we perform illusory patch localization for the dataset.

# Illusion generation:

We also generate illusions using text-to-image and image-to-image diffusion models. For generation, we use the diffusers library. 
The following notebook generates illusions using diffusion models:  illusion_diffusion.ipynb


# Instructions step by step:

Load the dataset and code:

a. Download dataset:
```
!wget https://www.cis.jhu.edu/~aroy/Supplementary_BRI3L.zip
```

b. unzip dataset: 
```
!unzip Supplementary_BRI3L.zip
```

# For illusion detection: (using PyTorch) (classify illusion vs non-illusion)
```
python Supplementary_BRI3L/Python_code_for_deep_models/illusion_classification_resnet_18.py
```
# For illusion vs natural image classification: (using PyTorch) 
```
python Supplementary_BRI3L/Python_code_for_deep_models/illusion_natural_img_resnet_18.py
```
# For illusion localization: (using keras and tensorflow) 
```
python Supplementary_BRI3L/Python_code_for_deep_models/unet_illusion_localization.py
```
# Illusion localization test on single image: (using keras and tensorflow) 
```
python Supplementary_BRI3L/Python_code_for_deep_models/test_on_single_img.py
```
# Location:

Location for illusion identification task: Supplementary_BRI3L/BRI3L_dataset/detection/new_illusion_classification/
Location for illusion localization task: Supplementary_BRI3L/BRI3L_dataset/Localization/ill_Loc/
Location for the learned model for illusion detection: Supplementary_BRI3L/Python_code_for_deep_models/resnet_illusion_detection.t7
Location for the learned model for illusion localization: Supplementary_BRI3L/Python_code_for_deep_models/model-tgs-salt_SSIM_epoch_25_diff_train_val.h5

We have provided a demo to load and run the dataset with our trained model and also exhibits some illustrations and visualization of the illusions and its predictions in the colab notebook: https://colab.research.google.com/drive/1g4Ov5Cbx4nIzd-QxabmtuFC9A-rMdrO0#scrollTo=MZbaM0Cn05MK
