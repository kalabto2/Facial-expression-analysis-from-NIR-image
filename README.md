# Facial Expression Analysis from NIR image

***

> This branch contains: code for mobilenet development and models checkpoints, exports and tests.

## Content
* *annotations_splits/* - folder containing data CSV annotations and other data for images in available databases
* *optuna_studies/* - this folder stores data for optua studies
* *tests/* - this folder contains test metrics and test metric images
* *data/* - empty folder where all ata should be placed
* *models/fer/* - contains models checkpoints and onnx exports of mobilenet.
* *.gitignore*
+ *BPutils.py* - this file contains helper utilities
+ *confusion_matrix_pretty_print.py*
* *README.md*
* *requirements.txt*
* *testing.py*
* *training.py*


## Set up
Setup virtual environment and install dependencies.
```
virtualenv venv
source venv/bin/activate
pip install -r requirements
```

Also, folder *data/* should contain necessary data.

### Used databases 
* AffectNet
* OuluCasia
* CASIA2.0
* BUAA
* Custom_DB

