import numpy as np
import SimpleITK as sitk


def normalize_image(processed_images_intensities, corrected_image, fat_reference_value=121, muscle_reference_value=40, output_image_path=None):
    """
    Normalize the corrected 3D image using linear scaling based on the 90th percentile for fat and
    the 10th percentile for muscle.

    Parameters:
        processed_images_intensities (numpy.ndarray): An array contains all the intensites
          under the post-processed area under bounding box for all images each class separtly.
        corrected_image (SimpleITK.Image): The 3D image after correction and post-processing.
        fat_reference_value (float, optional): The reference value for fat after normalization.
                                               Default is 121.
        muscle_reference_value (float, optional): The reference value for muscle after normalization.
                                                  Default is 40.
        output_image_path (str, optional): The path to save the normalized image. If not provided,
                                           the function will not save the image.

    Returns:
        normalized_image (SimpleITK.Image): The normalized 3D image.
    """
    # Calculate the 90th percentile for fat and the 10th percentile for muscle
    fat_intensity = np.percentile(processed_images_intensities['fat'], 90)
    muscle_intensity = np.percentile(
        processed_images_intensities['muscle'], 10)

    # Linearly scale the corrected image to the fat and muscle reference values
    normalized_image = ((corrected_image - muscle_intensity) / (fat_intensity - muscle_intensity)
                        ) * (fat_reference_value - muscle_reference_value) + muscle_reference_value

    # Write the normalized image to the output path if provided
    if output_image_path:
        sitk.WriteImage(normalized_image, output_image_path)

    # Return the normalized image
    return normalized_image
