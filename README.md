# Facial Expression Analysis from NIR image

***

> This branch contains a merged two branches - mobilenet and baseline. Thus all code (except unused code).

## Content
* *annotations_splits/* - folder containing data CSV annotations and other data for images in available databases
* *Custom_DB/* - contains code, images and annotations of *CUSTOM_DB*
	* *annotator_gui.py*, *annotator_gui_EN.py* - GUI annotator of custom data
+ *CustomMorphSet/* - contains code, setup guide and images of *CustomMorphSet*
* *data/* - empty folder where all data should be placed. If does not exist, create one.
* *example_data/* - empty folder where all example data should be placed. If does not exist, create one.
* *models/* - folder contains trained models in *onnx* format
	*  *models/face_detection/centerface.onnx*
	*  *models/spectrum_translation/* - models for spectrum translation
	*  *models/fer/* - models for facial expression analysis
* *notebooks/* - folder contains exploratory notebooks
* *optuna_studies/mobilenet_study-pretrained_on-affectnet_nir.pkl* - sotered Optuna studies
* *skeleton/* - contains source code for inference of models - a Python module called FaceInference
* *benchmark-face_detection.ipynb* - notebook for benchmark of face detection
* *benchmark-spectrum_translation.ipynb* - benchmark for image spectrum translation
* *BPutils.py* - helper file
* *CITATION.cff*
* *confusion_matrix_pretty_print.py* - renders confusion matrix
* *creating_splits.ipynb* - notebook for creating splits, setting up annotations
* *inference.ipynb* - notebook contains examples of inference usage
* *LICENSE*
* *README.md*
* *requirements.txt*
* *testing.ipynb* - Notebook for testing of MobileNet and DDAMFN.
* *training.ipynb* - Notebook for training of MobileNet.
* *.gitignore*

## Set up
Setup virtual environment and install dependencies.
```
virtualenv venv
source venv/bin/activate
pip install -r requirements
```

Also, folder *data/* or *example_data/* should contain necessary data.

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

### Demonstrative video
When recieved folder `example_data`, it also contains example video (`example_data/output1.mp4`) detecting and classifying faces from source viceo (`multiemotions_example_doubled.mp4`). 
This is an output of model from *Experiment2.2* thus the one trained on true NIR data. Other models can be also tested - follow examples from `inference.ipynb` such as *Experiment2.1.0* - the one trained on artificial data and the best model overall.


## Other code
Code for other networks were used from external sources and thus only mentioned here:
* CycleGAN - from repository from [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
* DDAMFN - from repository [https://github.com/simon20010923/DDAMFN](https://github.com/simon20010923/DDAMFN)

Code from custom CycleGAN is in branch *custom_models*, however, presented models were not trained on this code

## Used databases
* AffectNet
* OuluCasia
* CASIA2.0
* BUAA
* Custom_DB
* CustomMorphSet

Databases and other data can be requested via email [kalabis.tom@gmail.com](kalabis.tom@gmail.com).
