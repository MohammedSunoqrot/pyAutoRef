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
pyAutoRef/
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

## Diviation from the original published, MATLAB-based method
**`VERSION 1.0.0, 1.0.1, 1.0.2, 1.0.3, 1.0.4`**
- YOLOv8 model for object detector trained using the same data and bounding boxes used to train ACF in the original/MATLAB code. *No Data deviation, but needed to meantioned*. 
- The detection was performed on all slices for both fat and muscle.

**`VERSION 2.0.0`**
- YOLOv8 model for object detector trained using images from 823 cases (4 centers, 3 centers data is private and 1 is public which is the PROSTATE158 training dataset) in addition to the same data used to train the original detector. 
- For detection for both fat and muscle the first 15% and the last 15% of slices were not looked at for detection.

# How to cite AutoRef/pyAutoRef
In case of using or refering to AutoRef/pyAutoRef, please cite it as:
```
Sunoqrot, M.R.S., Nketiah, G.A., Selnæs, K.M. et al. Automated reference tissue normalization of T2-weighted MR images of the prostate using object recognition. Magn Reson Mater Phy 34, 309–321 (2021). [https://doi.org/10.1007/s10334-020-00871-3]
```

# How to use pyAutoRef
To perform AutoRef normalization, you first need to import the `autoref` function.
You can do it by calling `from pyAutoRef import autoref`

## `autoref` Function 
   - Parameters:
        input_image_path (str): The file path to the input 3D image (any supported SimpleITK format) or to the DICOM folder.
        output_image_path (str, optional): The file path to save the normalized output image to any supported SimpleITK format.
                                           If None, the image will not be saved.

   - Returns:
        normalized_image (SimpleITK.Image): The normalized 3D image.

## Supported input/output formats
- DICOM Series.
- All the medical [images formats supported by SimpleITK](https://simpleitk.readthedocs.io/en/v2.2.0/IO.html).
- [SimpleITK.Image] (https://simpleitk.org/SimpleITK-Notebooks/01_Image_Basics.html).

***DICOM Series is recognized when there is no file extension***

### Examples of usage:

***Example (input: medical image format, output: medical image format):***
```
from pyAutoRef import autoref

input_image_path = r"C:\Data\Case10_t2.nii.gz"
output_image_path = r"C:\Data\Case10_t2_normalized.nii.gz"

autoref(input_image_path, output_image_path)
```

***Example (input: medical image format, output: DICOM Series):***
```
from pyAutoRef import autoref

input_image_path = r"C:\Data\Case10_t2.nii.gz"
output_image_path = r"C:\Data\Case10_t2_normalized"

autoref(input_image_path, output_image_path)
```

***Example (input: DICOM Series, output: medical image format):***
```
from pyAutoRef import autoref

input_image_path = r"C:\Data\Case10_t2"
output_image_path = r"C:\Data\Case10_t2_normalized.nii.gz"

autoref(input_image_path, output_image_path)

```
***Example (input: DICOM Series, output: DICOM Series):***
```
from pyAutoRef import autoref

input_image_path = r"C:\Data\Case10_t2"
output_image_path = r"C:\Data\Case10_t2_normalized"

autoref(input_image_path, output_image_path)
```
