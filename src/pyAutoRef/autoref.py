import os
import time
import shutil
import pkg_resources
import logging

from pyAutoRef.pre_processing import pre_processing
from pyAutoRef.object_detection import object_detection
from pyAutoRef.post_processing import post_process_predictions
from pyAutoRef.normalization import normalize_image
from pyAutoRef.utils import save_image, check_predictions, check_input_image, get_intensities_without_detection


"""
This is the python version of the:
"Automated reference tissue normalization of T2-weighted MR images of the prostate using object recognition".
This is an automated method for dual-reference tissue (fat and muscle) normalization of T2-weighted MRI for the prostate.
Developed at the CIMORe group at the Norwegian University of Science and Technology (NTNU) in Trondheim, Norway. https://www.ntnu.edu/isb/cimore
For detailed information about this method, please read our paper: https://link.springer.com/article/10.1007%2Fs10334-020-00871-3

AUTHOR = 'Mohammed R. S. Sunoqrot'
AUTHOR_EMAIL = 'mohammed.sunoqrot@ntnu.no'
LICENSE = 'MIT'
GitHub: https://github.com/MohammedSunoqrot/pyAutoRef
"""

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def autoref(input_image, output_image_path=None):
    """
    autoref function for the AutoRef pipeline. This function takes the input image, performs
    pre-processing, object detection, post-processing, and normalization steps.

    Parameters:
        input_image (SimpleITK.Image, str): The input image as SimpleITK.Image OR The file path to the input 3D image (any supported SimpleITK format) or to the DICOM folder.
        output_image_path (str, optional): The file path to save the normalized output image to any supported SimpleITK format.
                                           If None, the image will not be saved.

    Returns:
        normalized_image (SimpleITK.Image): The normalized 3D image.

    Note:
        The normalized 3D image is saved to the specified output_image_path if provided.
    """
    # Measure the time taken for processing
    start_time = time.time()

    # Check if the input image valid
    try:
        input_image_type = check_input_image(input_image)
        logging.info(f"Input image type: {input_image_type}")
    except ValueError as e:
        logging.error(f"Invalid input image: {e}")
        return None

    # Print that the method started processing
    if input_image_type == 'Path':
        logging.info(
            f"=> Started AutoRef (fat and muscle) normalizing: {input_image}")
    else:
        logging.info(f"=> Started AutoRef (fat and muscle) normalizing...")

    # Get the current script file path
    current_file_path = os.path.abspath(__file__)

    # Get the directory (folder) path of the current script
    current_folder_path = os.path.dirname(current_file_path)

    # Perform pre-processing on the input image
    is_dicom, temp_images_dir, corrected_image, resized_corrected_image = pre_processing(
        current_folder_path, input_image)

    # Perform object detection on the preprocessed image
    model_path = pkg_resources.resource_filename(__name__, "model.onnx")
    top_predictions = object_detection(temp_images_dir, model_path)

    # Initial check for classes with zero predictions
    class_with_zero_predictions = check_predictions(top_predictions)

    # If any class had zero predictions, recalculate with new parameters
    if class_with_zero_predictions:
        logging.info(
            f"No detected objects for {class_with_zero_predictions}. Recalculating predictions...")

        # Perform object detection again with new parameters
        top_predictions = object_detection(temp_images_dir, model_path, yolo_classes=[
                                           "fat", "muscle"], slice_percent=[0, 1])

        # Check again after recalculating
        class_with_zero_predictions = check_predictions(top_predictions)

    # If still any class has zero predictions, normalize without detection
    if class_with_zero_predictions:
        logging.info(
            "Objects still not detected. Recalculating intensities without detection...")
        processed_images_intensities = get_intensities_without_detection(
            resized_corrected_image)
    else:
        # Perform post-processing to the detected objects
        processed_images_intensities = post_process_predictions(
            resized_corrected_image, top_predictions)

    # Perform normalization
    normalized_image = normalize_image(
        processed_images_intensities, corrected_image)

    # Write the normalized image to the output path if provided
    if output_image_path:
        save_image(normalized_image, input_image, is_dicom, output_image_path)

    # Delete the temp folder
    shutil.rmtree(temp_images_dir)

    # Measure the time taken for processing
    end_time = time.time()
    processing_time = end_time - start_time
    logging.info("==> Done with AutoRef (fat and muscle) normalizing. It took: {:.2f} seconds.".format(
        processing_time))
    if output_image_path:
        logging.info(f"   Output saved in: {output_image_path}")

    # Return the AutoRef normalized image
    return normalized_image
