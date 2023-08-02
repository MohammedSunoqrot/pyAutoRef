import SimpleITK as sitk

from pyAutoRef.utils import generate_random_temp_folder, read_sitk_image, read_dicom_folder, perform_n4_bias_field_correction, rescale_image, resize_image, write_slices_to_disk


def pre_processing(base_path, input_image_path,
                   scaling_method='percentile', scaling_method_args=[99, 100/99],
                   new_size=(384, 384), new_spacing=(0.5, 0.5)):
    """
   Pre-process the input image to prepare it to be used later.

    Parameters:
        base_path (str): The path to the main folder.
        input_image_path (str): The path to the input image file.
        scaling_method (str): The chosen scaling method ('none', 'max', 'median', or 'percentile'). Default is 'percentile'.
        scaling_method_args (list or tuple, optional): Additional arguments for the scaling method. Default is [99, 100/99].
        new_size (tuple, optional): The new size of the image in (rows, cols). Default is (384, 384).
        new_spacing (tuple, optional): The new pixel spacing in (rows, cols) in mm. Default is (0.5, 0.5).

    Returns:
        temp_dir_path (str): The generated temp folder path.
        corrected_image (SimpleITK.Image): The N4 bias field corrected image.
        resized_corrected_image (SimpleITK.Image): The N4 bias field corrected image after
          resized to new grid: 384 x 384 pixels of 0.5 x 0.5 mm.
    """
    # Folder for output images
    temp_images_dir = generate_random_temp_folder(base_path)

    # Read input image
    # Check if the input path corresponds to a DICOM directory.
    # If it is DICOM directory then "read_dicom_folder" function will be used.
    # Otherwise "read_sitk_image" function will be used.
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(input_image_path)

    if len(series_ids) > 0:
        # Input path contains DICOM data, use process_dicom function
        origial_image = read_dicom_folder(input_image_path)
    else:
        # Input path is another image format, use read_sitk_image function
        origial_image = read_sitk_image(input_image_path)

    # Apply ITK N4 bias field correction
    corrected_image = perform_n4_bias_field_correction(origial_image)

    # Rescale the corrected image: to the 99th percentile intensity value
    rescaled_image = rescale_image(
        corrected_image, scaling_method, scaling_method_args)

    # Resize fixed image to new grid: 384 x 384 pixels of 0.5 x 0.5 mm
    resized_image = resize_image(
        rescaled_image, new_size, new_spacing)

    # Write all the slices from the 'resized_image' to disk in the output directory
    write_slices_to_disk(resized_image, temp_images_dir)

    # Resize the corrected image to return to be used later
    resized_corrected_image = resize_image(
        corrected_image, new_size, new_spacing)

    # Return the temporary images folder path, the original size N4 bias field corrected image,
    # and the resized corrected image
    return temp_images_dir, corrected_image, resized_corrected_image
