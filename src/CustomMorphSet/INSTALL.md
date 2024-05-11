# Installation
*** 
1. clone content of the following repository into this folder: https://github.com/valillon/FaceMorph
2. install dependencies with `./install/install_morphing_dependencies_ubuntu.sh` (or install_morphing_dependencies_macos.sh)
3. install other dependencies:
```
pip install logging
```
4. insert images that will be morphed together into folder `original_images/`
5. set up the settings in `morph_customdb_nir.py`
5. run with `./morph_customdb_nir.py`