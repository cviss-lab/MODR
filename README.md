<img align="left" src="misc/logo.jpg">

### [CVISS Research](http://www.cviss.net/)



## Multi-output Image Classification to Support Post-Earthquake Reconnaissance

### Introduction

This repository contains the source code used by the authors for training and validating the Multi-Output Disaster Reconnaissance (MODR) model developed for the [paper]() (Under review),

Park, JA, Dyke, SJ, Yeum, CM, Midwinter, X, Liu, X, Choi, J, Sim, C, Chu, Z, Hacker, T, Benes, B. Multi -output Image Classification to Support Post-Earthquake Reconnaissance. *Journal of Performance of Constructed Facilities*. *Under review*.

MODR is a multi-output network trained on a wide variety of images taken during a typically post-earthquake field inspection. In this work, a multilevel hierarchical schema is developed to support many different types of post-earthquake reconnaissance missions. To allow rapid image classification under this schema, a multi-output image classification model was developed.

Once trained, the network can be used to rapidly categorize images into one or more relevant categories. This model is used to support rapid building report generation which aids engineers and planners to see the overall condition of the building in an organized manner.

This repository strictly deals with the training of the MODR network. 

### Dependencies

MODR was built using the following dependencies (**requirements.txt**).

Python version: **3.7.7**

```
numpy==1.18.4
tqdm==4.46.0
opencv_contrib_python==4.5.1.48
pandas==1.0.3
matplotlib==3.1.3
seaborn==0.10.1
tensorflow_gpu==2.2.0
Keras==2.4.3
scikit_learn==0.24.1
```

**NOTE** for training with a cpu, use tensorflow instead of tensorflow-gpu

## Training a image scale estimation model

#### unzip the dataset into the "datasets" folder

The data (**dataset.7z**) is located on the CVISS drive (**CVISS/Research_works/MODR/dataset.7z**) and is available upon request. Please send an email to Dr. Yeum (cmyeum@uwaterloo.ca).

#### Training configuration

In "**model_training.py**", the **config** (list of dict) variable contains the list of training configurations. Each dictionary inside the list specifies instructions used to train a single model. In this work, three different configurations were trained, and their configurations are shown as a example here.

```python
configs=[
    {  # sigmoid - single dense layer
        "epochs": 30,
        "output_pth": '../output/V6_efficientnet_single_dense_layer_sigmoid',
        "cw_setting": 'ml',
        'model_setting': 'efficientnet',
        "learning_rate": 0.01,
        "pth_to_data": "../datasets/V5/unique_dataset",
        "pth_to_labels": "../datasets/V5/multilabels_V2_moml.csv",
        "lf_setting": 'weighted_loss',
        "model_img_width": 299,
        "model_img_height": 299,
        "top_layer_option": "basic",
        'm_setting': ['acc', 'fbeta', 'recall', 'precision', 'specificity'],
    },
    {  # all softmax
        "epochs": 30,
        "output_pth": '../output/V6_efficientnet_all_softmax',
        "cw_setting": 'mc',
        'model_setting': 'efficientnet',
        "learning_rate": 0.01,
        "pth_to_data": "../datasets/V5/unique_dataset",
        "pth_to_labels": "../datasets/V5/multilabels_V2_moml.csv",
        "lf_setting": 'weighted_categorical_crossentropy',
        "model_img_width": 299,
        "model_img_height": 299,
        "top_layer_option": "multioutput",
        "af_setting": "softmax",
        'm_setting': ['acc', 'fbeta', 'recall', 'precision', 'specificity'],
    },
    {  # softmax - single dense layer
        "epochs": 30,
        "output_pth": '../output/V6_efficientnet_single_dense_layer_softmax',
        "cw_setting": 'mc',
        'model_setting': 'efficientnet',
        'af_setting': 'softmax',
        "learning_rate": 0.001,
        "pth_to_data": "../datasets/V5/unique_dataset",
        "pth_to_labels": "../datasets/V5/multilabels_V2_moml.csv",
        "lf_setting": 'weighted_categorical_crossentropy',
        "model_img_width": 299,
        "model_img_height": 299,
        "top_layer_option": "basic",
        'm_setting': ['acc', 'fbeta', 'recall', 'precision', 'specificity']
    }
]
```

        epochs (int): Number of times to cycle through the entire training dataset
        output_pth (str): Folder path to store model results
        cw_setting (str): One of [auto, mc, ml].
        	'auto': the keras class weighting auto setting
        	'mc': class weighting for multiclass classifier
        	'ml': class weighting for multilabel classifier
        model_setting (str): One of [xception, efficientnet, vgg16, resnetV2, mnetV2, densenet201] architectures
        af_setting (str): One of [softmax, sigmoid]. Softmax for multiclass, and sigmoid for multilabel.
        learning_rate (float): Controls the degree of change of weights during optimization
        pth_to_data (str): Path to the folder containing all the images
        pth_to_labels (str): Path to the csv file with the image filepaths and their corresponding labels
        lf_setting (str): One of [binary_crossentropy, categorical_crossentropy, weighted_categorical_crossentropy, weighted_loss]
        	'binary_crossentropy': Multilabel loss
        	'categorical_crossentropy': Multiclass loss
        	'weighted_categorical_crossentropy': Weighted multiclass loss
        	'weighted_loss': Weighted multilabel loss
        model_img_width (int): Width of model input
        model_img_height (int): Height of model input
        top_layer_option (str): One of [multioutput, basic]
        	'multioutput': specifies the multioutput structure
        	'basic': specifies a non-multioutput structure
        m_setting (list): List of relevant metrics to keep track of.
        image_augmentations (dict): Contains key-value pairs of augmentations corresponding to the augmentations in keras's ImageDataGenerator. By default there are minor brightness, width, height, zoom, and rotation augmentations, and horizontal flip augmentation.
        batch_size (int): Number of images per batch

**NOTE:** Model results of the three configurations are available upon request (**CVISS/Research_works/MODR/models.7z**). Please send an email to Dr. Yeum (cmyeum@uwaterloo.ca).

All model training results will be output in the **output** folder, which contains:

- model.h5: trained model
- train_hist.pkl: Training history
- training_config.pkl: training configuration used
- results: folder containing image results of patches
- train.csv and valid.csv: contains the actual labels for training and testing images
- pred.csv: model predictions for testing images
- pred.csv: raw model probabilities for testing images
- Variety of loss/accuracy metric history line plots/confusion matrix plot
- classification_report: summary of model performance in terms of precision, recall, and f1-score.

## Acknowledgements

Images from field reconnaissance missions from the following references were used for this work:

- Purdue University; NCREE (2016), "Performance of Reinforced Concrete Buildings in the 2016 Taiwan (Meinong) Earthquake," https://datacenterhub.org/resources/14098.
- Chungwook Sim; Enrique Villalobos; Jhon Paul Smith; Pedro Rojas; Santiago Pujol; Aishwarya Y Puranam; Lucas Laughery (2016), "Performance of Low-rise Reinforced Concrete Buildings in the 2016 Ecuador Earthquake," https://datacenterhub.org/resources/14160.
- Purdue University (2018), "Buildings Surveyed after 2017 Mexico City Earthquakes," https://datacenterhub.org/resources/14746.
- Nathaniel Sedra; Marc Eberhard; Ayhan Irfanoglu; Adolfo Matamoros; Santiago Pujol; Olafur Sveinn Haraldsson; David Alan Lattanzi; Steve Laurence Lauer; Bob Lyon; Josh Messmer; Kari Nasi; Jeffrey Rautenberg; Steeve Symithe; and Roby Douilly. (2017), "NEES: The Haiti Earthquake Database," https://datacenterhub.org/resources/263.
- Prateek Shah; Santiago Pujol; Aishwarya Puranam (2015), "Database on Performance of High-Rise Reinforced Concrete Buildings in the 2015 Nepal Earthquake," https://datacenterhub.org/resources/242.
- Prateek Shah; Santiago Pujol; Aishwarya Puranam; Lucas Laughery (2015), "Database on Performance of Low-Rise Reinforced Concrete Buildings in the 2015 Nepal Earthquake," https://datacenterhub.org/resources/238.
- Mete Sozen; Santiago Pujol; Ayhan Irfanoglu; Chungwook Sim; Aishwarya Y Puranam; Lucas Laughery; Suk Seung; Madeline Nelson; Merve Usta (2015), "Earthquake Reconnaissance Slides by Drs. Mete A. Sozen and Nathan M. Newmark," https://datacenterhub.org/resources/240.

## BibTeX Citation

```
TBD
```
