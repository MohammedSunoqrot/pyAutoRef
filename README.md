# pyAutoRef
This is the python version of the
"Automated reference tissue normalization of T2-weighted MR images of the prostate using object recognition"

This is an automated method for dual-reference tissue (fat and muscle) normalization of T2-weighted MRI for the prostate.

The method was developed at the CIMORe group at the Norwegian University of Science and Technology (NTNU) in Trondheim, Norway.
[https://www.ntnu.edu/isb/cimore]

For detailed information about this method, please read our paper: [https://link.springer.com/article/10.1007%2Fs10334-020-00871-3]

# Note
The provided algorithm was developed for research use and was NOT meant to be used in clinic.

# Structure
```
packaging_tutorial/
├── LICENSE
├── pyproject.toml
├── README.md
├── setup.cfg
├── src/
│   └── pyAutoRef/
│       ├── __init__.py
│       ├── autoref.py
│       ├── pre_processing.py
│       ├── object_detection.py
│       ├── post_processing.py
│       ├── normalization.py
│       ├── utils.py
│       ├── MANIFEST.in
│       └── model.onnx
└── tests/
```

# Installation
You can install the package either from pip or using pip or the files in GitHub repository [https://github.com/MohammedSunoqrot/pyAutoRef]

## pip
Simply type:
```
pip install pyAutoRef
```
## GitHub
- Clone the GitHub repository
   *From command line*
   ```
   git clone https://github.com/MohammedSunoqrot/pyAutoRef.git
   ```
- Change directory to the clones folder (unzip if needed) and type
   ```
   pip install . 
   ```

# Difference Note
This python version is differ than the originally published MATLAB version [https://github.com/ntnu-mr-cancer/AutoRef] in terms:
- It accepts all kind of SimpleITK supported image format.
- It uses YOLOv8 model for object detector instead of ACF.
   (This model is trained using the same data and bounding boxes used to train ACF in the original/MATLAB code).

# How to cite AutoRef/pyAutoRef
In case of using or refering to AutoRef/pyAutoRef, please cite it as:
```
Sunoqrot, M.R.S., Nketiah, G.A., Selnæs, K.M. et al. Automated reference tissue normalization of T2-weighted MR images of the prostate using object recognition. Magn Reson Mater Phy 34, 309–321 (2021). [https://doi.org/10.1007/s10334-020-00871-3]
```

# How to use pyAutoRef
There are two input parameters:
1- input_image_path (str): The file path to the input 3D image (any supported SimpleITK format).
2- output_image_path (str, optional): The file path to save the normalized output image to any supported SimpleITK format.
            If None, the image will not be saved.

***Example of usage:***
```
from pyAutoRef import autoref

input_image_path = r"C:\Data\Case10_t2.nii.gz"
output_image_path = r"C:\Data\Case10_t2_normalized.nii.gz"

autoref(input_image_path, output_image_path)
```
