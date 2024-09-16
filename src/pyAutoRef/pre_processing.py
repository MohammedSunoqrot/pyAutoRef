from pyAutoRef.utils import generate_random_temp_folder, read_sitk_image, perform_n4_bias_field_correction, rescale_image, resize_image, write_slices_to_disk, check_input_image


def pre_processing(base_path, input_image,
                   scaling_method='percentile', scaling_method_args=[99, 100/99],
                   new_size=(384, 384), new_spacing=(0.5, 0.5)):
    """
    Pre-process the input image to prepare it for further processing.

    Parameters:
        base_path (str): The path to the main folder.
        input_image (str): The input image variable/file/folder.
        scaling_method (str): The chosen scaling method ('none', 'max', 'median', or 'percentile'). Default is 'percentile'.
        scaling_method_args (list or tuple, optional): Additional arguments for the scaling method. Default is [99, 100/99].
        new_size (tuple, optional): The new size of the image in (rows, cols). Default is (384, 384).
        new_spacing (tuple, optional): The new pixel spacing in (rows, cols) in mm. Default is (0.5, 0.5).

    Returns:
        is_dicom (bool): Is the input a DICOM folder.
        temp_dir_path (str): The generated temporary folder path.
        corrected_image (SimpleITK.Image): The N4 bias field corrected image.
        resized_corrected_image (SimpleITK.Image): The N4 bias field corrected image resized to new grid: 384 x 384 pixels of 0.5 x 0.5 mm.
    """
    # Generate a temporary folder for output images
    temp_images_dir = generate_random_temp_folder(base_path)

    # Check if the input image is a path or already an image object
    input_image_type = check_input_image(input_image)

    if input_image_type == 'Path':
        try:
            # Read the input image
            original_image, is_dicom = read_sitk_image(input_image)
        except Exception as e:
            raise RuntimeError(f"Error reading input image: {e}")
    else:
        original_image = input_image
        is_dicom = False

    # Apply ITK N4 bias field correction
    corrected_image = perform_n4_bias_field_correction(original_image)

    # Rescale the corrected image to the 99th percentile intensity value
    rescaled_image = rescale_image(
        corrected_image, scaling_method, scaling_method_args)

    # Resize the image to new grid: 384 x 384 pixels of 0.5 x 0.5 mm
    resized_image = resize_image(rescaled_image, new_size, new_spacing)

    # Write all the slices from the resized image to disk in the output directory
    write_slices_to_disk(resized_image, temp_images_dir)

    # Resize the corrected image for later use
    resized_corrected_image = resize_image(
        corrected_image, new_size, new_spacing)

    # Return the temporary images folder path, the original size N4 bias field corrected image,
    # and the resized corrected image
    return is_dicom, temp_images_dir, corrected_image, resized_corrected_image
