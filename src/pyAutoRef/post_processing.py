import numpy as np
import SimpleITK as sitk

from skimage import img_as_ubyte
from skimage.filters import threshold_otsu
from skimage.morphology import disk, opening, erosion

from pyAutoRef.utils import detect_remove_outliers, extract_largest_connected_component, extract_intensities_from_mask


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

    # Process each class prediction
    for class_name, predictions in top_predictions.items():
        class_intensities = []  # List to store intensities for the current class

        for prediction in predictions:
            # Convert slice number to integer for indexing
            slice_number = int(prediction['slice'])
            image_slice = image_3D_wo_outliers[:, :, slice_number]

            # Convert the SimpleITK Image to a NumPy array
            image_slice_np = sitk.GetArrayViewFromImage(image_slice)

            # Make a writable copy of the array
            image_slice_np = np.copy(image_slice_np)

            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, prediction['bbox'])

            # Crop the image to the bounding box
            cropped_image = image_slice_np[y1:y2, x1:x2]
            # Handle NaN values in the cropped image
            cropped_image = np.nan_to_num(cropped_image)

            # Debug: Check if cropped_image is empty
            if cropped_image.size == 0:
                print(f"Empty cropped image for prediction: {prediction}.")
                continue

            # Extract the area based on class (0: 'fat', 1: 'muscle')
            if class_name == 'fat':
                binary_image = img_as_ubyte(
                    cropped_image > threshold_otsu(cropped_image))
            else:
                binary_image = img_as_ubyte(
                    cropped_image < threshold_otsu(cropped_image))

            # Perform morphological opening with disk shape of opening_radius
            processed_image = opening(binary_image, disk(opening_radius))

            # Perform morphological erosion with disk shape of erosion_radius
            processed_image = erosion(processed_image, disk(erosion_radius))

            # Extract the largest connected component from the binary image
            largest_component = extract_largest_connected_component(
                processed_image)

            # Overlay the mask on the image slice and extract intensities
            masked_intensities = extract_intensities_from_mask(
                cropped_image, largest_component.astype(bool))

            # Append the intensities to the list for the current class
            class_intensities.extend(masked_intensities)

        # Convert the list of intensities to a NumPy array and store in the dictionary
        processed_images_intensities[class_name] = np.array(class_intensities)

    # Return the intensities arrays of the detected classes objects
    return processed_images_intensities
