import numpy as np
import SimpleITK as sitk


def normalize_image(processed_images_intensities, corrected_image, fat_reference_value=121, muscle_reference_value=40,
                    fat_intensity_percentile=95, muscle_intensity_percentile=5):
    """
    Normalize the corrected 3D image using linear scaling based on the specified percentiles for fat and muscle intensities.

    This function performs normalization of the 3D image by calculating the intensity values for fat and muscle
    based on specified percentiles and then scaling the corrected image accordingly.

    Parameters:
        processed_images_intensities (dict): A dictionary containing arrays of intensities extracted from post-processed areas.
                                             Keys are class names ('fat', 'muscle') and values are arrays of intensities.
        corrected_image (SimpleITK.Image): The 3D image that has been corrected and post-processed.
        fat_reference_value (float, optional): The target reference value for fat after normalization. Default is 121.
        muscle_reference_value (float, optional): The target reference value for muscle after normalization. Default is 40.
        fat_intensity_percentile (int, optional): The percentile value to determine fat intensity. Default is 95.
        muscle_intensity_percentile (int, optional): The percentile value to determine muscle intensity. Default is 5.

    Returns:
        SimpleITK.Image: The normalized 3D image.
    """
    # Extract the intensity values for fat and muscle
    fat_intensities = processed_images_intensities.get('fat', np.array([]))
    muscle_intensities = processed_images_intensities.get(
        'muscle', np.array([]))

    # Calculate intensity percentiles
    fat_intensity = np.percentile(
        fat_intensities, fat_intensity_percentile) if fat_intensities.size > 0 else 1
    muscle_intensity = np.percentile(
        muscle_intensities, muscle_intensity_percentile) if muscle_intensities.size > 0 else 0

    # Convert SimpleITK image to NumPy array for processing
    corrected_image_np = sitk.GetArrayFromImage(corrected_image)

    # Apply linear normalization
    normalized_image_np = ((corrected_image_np - muscle_intensity) / (fat_intensity - muscle_intensity)
                           ) * (fat_reference_value - muscle_reference_value) + muscle_reference_value

    # Convert the NumPy array back to SimpleITK image
    normalized_image = sitk.GetImageFromArray(normalized_image_np)
    normalized_image.CopyInformation(corrected_image)

    # Return the normalized image
    return normalized_image
