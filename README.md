# Facial Expression Analysis from NIR image

***

> This branch contains: trained spectrum translation models, inference code, Custom_DB and data related code.

## Content
* *annotations_splits/* - folder containing data CSV annotations and other data for images in available databases.
* *Custom_DB/* - contains code, images and annotations of *CUSTOM_DB*
	* *annotator_gui.py*, *annotator_gui_EN.py* - GUI annotator of custom data
+ *CustomMorphSet/* - contains code, setup guide and images of *CustomMorphSet*
* *data/* - empty folder where all data should be placed
* *models/* - folder contains trained models in *onnx* format
	*  *models/face_detection/centerface.onnx*
	*  *models/spectrum_translation/* - models for spectrum translation
	*  *models/fer/* - models for facial expression analysis
* *notebooks/* - folder contains exploratory notebooks
* *skeleton/* - contains source code for inference of models - a Python module called FaceInference
* *benchmark-face_detection.ipynb* - notebook for benchmark of face detection
* *benchmark-spectrum_translation.ipynb* - benchmark for image spectrum translation
* *creating_splits.ipynb* - notebook for creating splits, setting up annotations
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

## Installation of Python Module
Python module named FaceInference can be installed to pip with the following:
```bash
# when used virtual environment, switch to it (uncomment the line)
# source venv/bin/activate

# change to the directory and install the module
cd skleton
pip install .
```

Then the module can be imported with `import FaceInference` inside the python script.

## Example inference
The example inference notebook can be found in `inference.ipynb`.
When  was recieved also additional data with this evaluated notebook and other, place the `example_data/` folder to the root of this directory and replace the newly obtained `inference.ipynb` with the oni in this repository. 


## Used databases
* AffectNet
* OuluCasia
* CASIA2.0
* BUAA
* Custom_DB
* CustomMorphSet

Databases can be requested via email kalabis.tom@gmail.com
