## General environment for classification tasks using Tensorflow 2 and Keras APIs

#### Development

The code is following this naming conventions:

https://visualgit.readthedocs.io/en/latest/pages/naming_convention.html

#### General Information

Supported Classification models can be found here: [kerasAPI](https://keras.io/api/applications/).

Supported backbones include ResNet50/34/18, SEResNet50/34/18, ResNetX50, SEResNeXt50, DenseNet121, MobileNet,
MobileNetV2, NASNetMobile, Xception and EfficientNetB0 to EfficientNetB7. Additionally, CSPDarkNet53 and
CSPDarkNet53Tiny backbones, as well as Yolov4 and Yolov4Tiny heads are supported for detection.

#### Installation

Open a terminal, change directory to ```Classifier/``` and create conda environments using the following command lines:

```
conda env create -f ./src/environment_tf2.3.yml --name <env_name>
```

```
conda env create -f ./src/environment_tf2.4.yml --name <env_name>
```

**Note: for windows, tf2.3 works with no issues for all tasks. For Ubuntu, use tf2.4 for classification

The following packages and their dependencies will be installed:

| Package  | Version |
|-----|-----|
| python | 3.8.3| 
| tensorflow-gpu | 2.3.0 or 2.4.1| 
| cudatoolkit | 10.1.243 |
| cudnn | 7.6.5 |
| tensorflow-addons | 0.12.1 |
| imgaug | 0.4.0  |
| matplotlib | 3.3.2 |
| pandas | 1.1.3|
| albumentations|0.5.2|
| scikit-learn | 0.23.2|
| azureml-sdk | 1.29.0| 
| imagehash | 4.2.1| 
| tqdm | 4.62.2| 
| opencv-python | 4.2.0.34 |
| tf-keras-vis | 0.8.1 |
| optuna | 2.10.1 |

#### Usage

Using python files on a linux terminal or under Windows in an Anaconda terminal (or IDE terminal with Anaconda
interpreter). Run the run_task.py script with following arguments:
- ```--task```: target task. Task should be 'train', 'test' or 'interpret'
- ```--target```: name of compute instance to run the code. For passing 
        * 'train-no-deploy': train is local and not logged in Azure.
        * 'local': train is local and logged in azure.
        * A compute_unit_name: train is on cloud and logged
- ```--config```: path to config file, see samples in folder configs/

#### Datasets

Datasets should follow this structure:

- Classification

Class names and files' names are user defined, please keep the class numbering as shown

```
dataset/
        00_class_name/
                img_1.png
                img_2.png
                ...
        01_class_name/
                img_3.png
                img_4.png
                ...
        02_class_name/
                ...
```

- Multi-label Classification

img+gt/ directory contains both images and their corresponding ground truth in a binarized fashion, e.g. 1 0 1 1

```
dataset/
        img+gt/
            img_1.png
            img_1.txt
            img_2.png
            img_2.txt
            ...
```

---

#### Train task

Run the run_task.py script with 'train' as task argument and set the desired parameters from the table below
in a yaml file. See samples in folder configs/ 

| Parameters  | Description |
| ------------- | ------------- |
|--dataset|Path to the training dataset|
|--ouput|Path to the Output Directory for results|
|--valdata|Directory of validation data. If not defined, data will be randomly split into train+val |
|--validation_split|ratio of the data to be used as validation. Default is <0.05>|
|--backbone|Backbone architecture. Currently supporting ResNet50/34/18, SEResNet50/34/18, ResNetX50, SEResNeXt50, DenseNet121, MobileNet, MobileNetV2, NASNetMobile, Xception and EfficientNetB0-B7.
|--epochs|Number of training epochs. Default is set to 10 |
|--batch|Batch size. Default is set to 16 |
|--dim|Dimensions of input images after resizing. Can be either single value for square (dim x dim) or two values for (height x width) = (dim[0] x dim[1]). Default is set to <224>|
|--lr|(initial in case of decay) learning rate of the optimizer. Default is set to <0.001> |
|--lr_decay|Activate step learning rate decay. For detection, cosine lr decay is activated even when not specified |
|--dropout|Dropout rate to be enforced on the densely connected head of classifiers. for EfficientNets its fixed at 0.2 |
|--train_backbone|Update backbone parameters while training|
|--num_classes|Number of classes. If not given, <2> will be taken by default|
|--num_labels|Number of labels in a multi-label classification task.|
|--class_names|Define class names to include in args.json|
|--check_data|Check if all samples in dataset can be processed|
|--resize|Resizing method for classification and segmentation. Currently supporting center-crop, padding and downscale. Default is center-crop. Padding is fixed for detection.|
|--label_smoothing|Smooth labels for classification tasks to reduce overconfidence (experimental)|
|--bayesian|dropout-rate for monte carlo dropout layers added before each trainable layer|

Examples:

```
Run in terminal
python run_task.py --task train --target local --config ./configs/file_name.yml

- Binary classification with a fixed learning rate and small validation split 
Parameter: --dataset C:\...\smoke01-1628 --backbone ResNet50 --lr 0.0001 --validation_split 0.01

- Binary classification with a bayesian network and defined validation set
Parameter: --dataset C:\...\SmokeCholec80\train --valdata C:\...\SmokeCholec80\test --backbone EfficientNetB0 --bayesian 0.3
 
- Multiclass classification with step learning rate decay step starting at 0.01 using downscale resizing
Parameter: --dataset C:\...\InOut200830 --num_classes 9 --backbone MobileNet --lr_decay --lr 0.01 --resize downscale
  
- Multilabel classification with backbone training 
Parameter: --dataset C:\....\heichole --backbone EfficientNetB3 --train_backbone --num_classes 7

``` 

After training, the model, the model weights and training args.json file will be saved under `user-defined output/<timestamp>`

To track training progress using Tensorboard, open a new terminal in the project root folder, activate the environment
and type:

```
tensorboard --logdir ./logs/<timestamp>
```

Then open the given link in browser.

---

#### run_train_optuna.py

Run run_train_optuna.py script for hyperparemeter optimization.
The parameters to be optimised must be edited directly in the file: `train_optuna.py`. For more information about [optuna](https://optuna.org/)
Example:

```
Run in terminal
python run_train_optuna.py --target gpu-cluster --config ./configs/file_name.yml
```
---

#### Test task

Run the run_task.py script with 'test' as task argument and set the desired parameters from the table below 
in a yaml file. See samples in folder configs/

Parameters --backbone, --resize, --head, --num_classes, --num_labels and --dim will be overwritten by args.json

| Required parameters (all)  | Description |
| ------------- | ------------- |
|--dataset|Path to the test dataset|
|--ouput|Path to the Output Directory for results|
|--saved_model|Path to the saved model |
|--class_names|Define class names to include in plots and tables|


|Required parameter (only one) | Description                                                                                                 | Optional parameter                        | Description                                                                                                                                            |
|-------------------------------|-------------------------------------------------------------------------------------------------------------|-------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| --class_eval                  | Generate confusion matrix for classification predictions and print various evaluation metrics. GT required  | --class_names                             | Define confusion matrix axes labels                                                                                                                    |
|                               |                                                                                                             | --temp_filter                             | Activate weighted-mean temporal filtering, create a confusion matrix for filtered results and generate a line plot                                     |
|                               |                                                                                                             | --filter_size                             | Size of buffer for temporal filtering (default is <10>)                                                                                                |
|                               |                                                                                                             | --bayesian                                | dropout-rate for monte carlo dropout layers added before each trainable layer                                                                          |
|                               |                                                                                                             | --bayesian_samples                        | number of bayesian samples to evaluate (default is <10>)                                                                                               |
| --class_overlay               | Overlay classification samples with the model output. GT not required                                       | --class_names                             | Define class names to be displayed                                                                                                                     |
|                               |                                                                                                             | --temp_filter                             | Activate weighted-mean temporal filtering                                                                                                              |
|                               |                                                                                                             | --filter_size                             | Size of buffer for temporal filtering (default is <10>)                                                                                                |
| --class_sort                  | Sort classifier predictions according to class names. GT not required                                       | --class_names                             | Define names of generated repositories ,i.e. classes                                                                                                   |
|                               |                                                                                                             | --temp_filter                             | Activate weighted-mean temporal filtering                                                                                                              |
|                               |                                                                                                             | --filter_size                             | Size of buffer for temporal filtering (default is <10>)                                                                                                |
| --class_table                 | Generate a .csv file with sample names, ground truths (not required), labels and raw output                 | --class_names                             | Define names of generated repositories ,i.e. classes                                                                                                   | 
|                               |                                                                                                             | --temp_filter                             | Activate weighted-mean temporal filtering                                                                                                              |
|                               |                                                                                                             | --filter_size                             | Size of buffer for temporal filtering (default is <10>)                                                                                                |

**An optional --check_data parameter can be added to make sure all data samples can be processed successfully.

**If class_names should contain whitespaces, set the names in quotation marks

Examples:

```
Run in terminal
python run_task.py --task test --target local --config ./configs/file_name.yml

- Evaluate multi-class classifier 
Parameter: --class_eval --data /.../dataset --saved_model /models/model  --check_data --class_names class_0 class_1 class_3

- Evaluate bayesian multi-class classifier 
Parameter: --class_eval --data /.../dataset --saved_model /models/model --bayesian 0.3 --bayesian_samples 10 --check_data --class_names class_0 class_1 class_3

- Overlay model outputs over images for a binary classifier and save the results
Parameter: --class_overlay --data /.../dataset --saved_model /models/model --check_data

- Sort data samples according to classifier predictions
Parameter: --class_sort --data /.../dataset --saved_model /models/model

- Evaluate classification on sequence of images and apply weighted mean temporal filter
Parameter: --class_eval --data /.../dataset --saved_model /models/model --check_data --temp_filter --filter_size 9

``` 

---

#### Interpret task

Run the run_task.py script with 'interpret' as task argument to generate an interpretability plot. 
Set the desired parameters from the table below in a yaml file. See samples in folder configs/
Parameters --backbone, --resize, --head, --num_classes, --num_labels and --dim will be overwritten by args.json

| Required parameters (all)  | Description                                                                                                                         |
| ------------- |-------------------------------------------------------------------------------------------------------------------------------------|
|--dataset| Path to the test dataset                                                                                                            |
|--ouput|Path to the Output Directory for results                                                               |
|--saved_model| Path to the saved model                                                                                                             |
|--baseline| Baseline used in integrated gradients. Currently supporting 'black', 'white', 'random', 'mean'. Default is 'black'.|                 
|--interpreter| Interpretability method. Currently supporting 'ig', 'gradcam', 'gradcam++', 'smoothgrad'. Default is  'ig'.|
|--cmap| Colormap for heatmap. See [colormaps](https://matplotlib.org/stable/tutorials/colors/colormaps.html) for available colormaps in matplotlib. Default is inferno.|
|--overlay_alpha| Alpha value for overlay of heatmap and image. Default is 0.4|

Examples:

```
Run in terminal
python run_task.py --task interpret --target local --config ./configs/file_name.yml

- Integrated Gradients with black baseline (default)
Parameter: --dataset C:\...\dataset --saved_model /models/model 

- Integrated Gradients with white baseline
Parameter: --dataset C:\...\dataset --saved_model /models/model --baseline white

- Integrated Gradients with higher overlay alpha
Parameter: --dataset C:\...\dataset --saved_model /models/model --overlay_alpha 0.8

- GradCAM
Parameter: --dataset C:\...\dataset --saved_model /models/model --interpreter gradcam

- GradCAM++
Parameter: --dataset C:\...\dataset --saved_model /models/model --interpreter gradcam++

- GradCAM with 'hot' colormap
Parameter: --dataset C:\...\dataset --saved_model /models/model --interpreter gradcam --cmap hot

- SmoothGrad
Parameter: --dataset C:\...\dataset --saved_model /models/model --interpreter smoothgrad
