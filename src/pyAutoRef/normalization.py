import numpy as np

def normalize_image(processed_images_intensities, corrected_image, fat_reference_value=121, muscle_reference_value=40,
                    fat_intensity_percentile=95, muscle_intensity_percentile=5):
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
        fat_intensity_percentile (int, optional): The percentile for fat intensity to represent detected fat. Default is 95.
        muscle_intensity_percentile (int, optional): The percentile for muscle intensity to represent detected muscle. Default is 5.

    Returns:
        normalized_image (SimpleITK.Image): The normalized 3D image.
    """
    # Calculate the 95th/defined percentile for fat and the 5th/defined percentile for muscle
    fat_intensity = np.percentile(processed_images_intensities['fat'], fat_intensity_percentile)
    muscle_intensity = np.percentile(processed_images_intensities['muscle'], muscle_intensity_percentile)

    # Linearly scale the corrected image to the fat and muscle reference values
    normalized_image = ((corrected_image - muscle_intensity) / (fat_intensity - muscle_intensity)
                        ) * (fat_reference_value - muscle_reference_value) + muscle_reference_value

    # Return the normalized image
    return normalized_image
