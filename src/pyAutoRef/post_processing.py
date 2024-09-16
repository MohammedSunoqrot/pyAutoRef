import numpy as np
import SimpleITK as sitk
import logging
from skimage import img_as_ubyte
from skimage.filters import threshold_otsu
from skimage.morphology import disk, opening, erosion
from concurrent.futures import ThreadPoolExecutor, as_completed

from pyAutoRef.utils import detect_remove_outliers, extract_largest_connected_component, extract_intensities_from_mask

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def process_slice(slice_number, image_3D_wo_outliers, prediction, opening_radius, erosion_radius, class_name):
    """
    Take a slice number, extracts the relevant slice from the 3D image, applies thresholding and
    morphological operations (opening and erosion), and extracts the largest connected component to calculate
    the intensities within the bounding box specified by the prediction.

    Parameters:
        slice_number (int): The index of the slice to process.
        image_3D_wo_outliers (SimpleITK.Image): The 3D image with outliers removed.
        prediction (dict): A dictionary containing the 'slice' number (integer), the bounding box coordinates,
                            and the probability score.
        opening_radius (int): The radius of the disk used for morphological opening. Default is 5.
        erosion_radius (int): The radius of the disk used for morphological erosion. Default is 5.
        class_name (str): The class name ('fat' or 'muscle') for which the slice is being processed.

    Returns:
        masked_intensities (list): A list of intensities extracted from the post-processed area within the bounding box.
              Returns an empty list if the cropped image is empty.
    """
    image_slice = image_3D_wo_outliers[:, :, slice_number]

    # Convert the SimpleITK Image to a NumPy array
    image_slice_np = sitk.GetArrayViewFromImage(image_slice)
    image_slice_np = np.copy(image_slice_np)

    # Get the image dimensions
    height, width = image_slice_np.shape[:2]

    # Extract bounding box coordinates
    x1, y1, x2, y2 = map(int, prediction['bbox'])

    # Ensure x1, y1, x2, y2 are within image boundaries
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    # Crop the image to the bounding box
    cropped_image = image_slice_np[y1:y2, x1:x2]
    cropped_image = np.nan_to_num(cropped_image)

    if cropped_image.size == 0:
        logging.warning(f"Empty cropped image for prediction: {prediction}.")
        return []

    # Extract the area based on class
    if class_name == 'fat':
        binary_image = img_as_ubyte(
            cropped_image > threshold_otsu(cropped_image))
    else:
        binary_image = img_as_ubyte(
            cropped_image < threshold_otsu(cropped_image))

    # Perform morphological opening and erosion
    processed_image = opening(binary_image, disk(opening_radius))
    processed_image = erosion(processed_image, disk(erosion_radius))

    # Extract the largest connected component
    largest_component = extract_largest_connected_component(processed_image)

    # Overlay the mask on the image slice and extract intensities
    masked_intensities = extract_intensities_from_mask(
        cropped_image, largest_component.astype(bool))

    return masked_intensities


def post_process_predictions(image_3D, top_predictions, opening_radius=5, erosion_radius=5):
    """
    Post-process the top predictions for each class by taking the area under the bounding box,
    applying Otsu thresholding, performing morphological opening,
    extracting the largest connected component, and returning the masks to the original size.

    Parameters:
        image_3D (SimpleITK.Image): The 3D image containing the slices.
        top_predictions (dict): A dictionary containing the top 3 predictions for each class.
                                The keys are class names, and the values are lists of dictionaries.
                                Each dictionary contains the 'slice' number (integer),
                                the bounding box coordinates, and the probability score.
        opening_radius (int): The radius of the disk used for morphological opening. Default is 5.
        erosion_radius (int): The radius of the disk used for morphological erosion. Default is 5.

    Returns:
        processed_images_intensities (numpy.ndarray): An array contains all the intensites
          under the post-processed area under bounding box for all images each class separtly.
    """
    # Detect and remove outliers in the 3D image
    image_3D_wo_outliers = detect_remove_outliers(image_3D)

    # Initialize a dictionary to store the post-processed images for each class
    processed_images_intensities = {class_name: []
                                    for class_name in top_predictions}

    # Use ThreadPoolExecutor to parallelize slice processing
    with ThreadPoolExecutor() as executor:
        futures = {}

        # Submit tasks to the executor
        for class_name, predictions in top_predictions.items():
            for prediction in predictions:
                slice_number = int(prediction['slice'])
                future = executor.submit(process_slice, slice_number, image_3D_wo_outliers,
                                         prediction, opening_radius, erosion_radius, class_name)
                futures[future] = class_name  # Map future to class_name

        # Collect results as they complete
        for future in as_completed(futures):
            result = future.result()
            if result.any():
                # Retrieve class_name from the future
                class_name = futures[future]
                processed_images_intensities[class_name].extend(result)

    # Convert lists to NumPy arrays
    for class_name in processed_images_intensities:
        processed_images_intensities[class_name] = np.array(
            processed_images_intensities[class_name])
        logging.info(
            f"Processed intensities for class {class_name}: {processed_images_intensities[class_name].shape}")

    # Return the intensities arrays of the detected classes objects
    return processed_images_intensities
