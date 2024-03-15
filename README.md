# Facial Expression Analysis from NIR image

***

> This branch contains: trained spectrum translation models, inference code, Custom_DB and data related code.

## Content
* *annotations_splits/* - folder containing data CSV annotations and other data for images in available databases
* *Custom_DB/* - contains code, images and annotations of *CUSTOM_DB*
	* *annotator_gui.py*, *annotator_gui_EN.py* - GUI annotator of custom data
* *data/* - empty folder where all ata should be placed
* *models/* - folder contains trained models in *onnx* format
	*  *models/face_detection/centerface.onnx*
	*  *models/spectrum_translation/* - models for spectrum translation
* *notebooks/* - folder contains exploratory notebooks
* *skeleton/* - contains source code for inference of models
* *benchmark-face_detection.ipynb* - notebook for benchmark of face detection
* *benchmark-spectrum_translation.ipynb* - benchmark for image spectrum translation
* *creating_split.ipynb* - notebook for creating splits, setting up annotations
* *inference.ipynb* - notebook contains examples of inference usage
* *README.md*
* *requirements.txt*
* *.gitignore*

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

