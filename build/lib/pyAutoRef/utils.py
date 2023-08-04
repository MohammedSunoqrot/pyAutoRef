import os
import numpy as np
import SimpleITK as sitk
import pydicom
import uuid
import multiprocessing
import onnxruntime as ort
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
import logging

from pydicom.uid import generate_uid
from functools import wraps
from skimage import measure
from PIL import Image


def generate_random_temp_folder(base_path):
    """
   Generate a temp folder with uniqe random name.

    Parameters:
        base_path (str): The path to the main folder.

    Returns:
        temp_dir_path (str): The generated temp folder path.
    """
    # Create a temporary directory with a unique name for each parallel process
    temp_dir_name = f"temp_{uuid.uuid4().hex}"
    temp_dir_path = os.path.join(base_path, temp_dir_name)

    try:
        # Create the temporary directory
        os.makedirs(temp_dir_path)
    except FileExistsError:
        # If the directory already exists (unlikely due to the unique name),
        # retry with a different name
        return generate_random_temp_folder(base_path)

    return temp_dir_path

    """ 
    # Example usage:
    base_path = "/path/to/your/base/directory"
    temp_folder = generate_random_temp_folder(base_path)
    print("Random temporary folder path:", temp_folder)
     """


def read_sitk_image(file_path):
    """
    Reads an image file (with supported SimpleITK format) and returns a SimpleITK image object.

    Parameters:
        file_path (str): The path to the file/folder(for DICOM).

    Returns:
        image (SimpleITK.Image): The SimpleITK image object representing the image.
        is_dicom (bool): Is the input a DICOM folder. 
    """

    # Check if the input path corresponds to a DICOM directory.
    # Check if the output_file_path has no file extension
    if not os.path.splitext(file_path)[1]:  
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(file_path)

        if len(series_ids) > 0:
            # It is DICOM directory
            # Get the DICOM files and then read them as a 3D image
            dicom_names = reader.GetGDCMSeriesFileNames(file_path)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            # Return it is DICOM
            is_dicom = True
        else:
            # It is not DICOM directory
            # Read the image
            image = sitk.ReadImage(file_path)
            # Return it is not DICOM
            is_dicom = False
    else:
            # It is not DICOM directory
            # Read the image
            image = sitk.ReadImage(file_path)
            # Return it is not DICOM
            is_dicom = False

    # Check if the image is not of floating-point type
    if image.GetPixelID() not in (sitk.sitkFloat32, sitk.sitkFloat64):
        # Convert image to Float32 pixel type
        image = sitk.Cast(image, sitk.sitkFloat32)

    return image, is_dicom


def sitk_image_to_dicom_series(image, input_file, output_folder, is_dicom=False):
    """
    Convert a SimpleITK Image to a DICOM series and save it to the specified output folder.

    Parameters:
        image (SimpleITK.Image): The input 3D  SimpleITK Image to be converted to DICOM.
        input_file (str): The path to the file/folder(for DICOM).
        output_folder (str): The path to the output folder where the DICOM files will be saved.
        is_dicom (bool): If True, copy DICOM tags from the input dicom_series to the output DICOM series.

    Returns:
        None.
    """
    # If the output folder not exists make it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Check if the input file/folder is DICOM series 
    if is_dicom:
        if not os.path.isdir(input_file):
            raise ValueError("Input file should be a folder containing DICOM files when is_dicom is True.")
        
        # Load meta tags from the input DICOM series
        input_dicom_files = [os.path.join(input_file, f) for f in os.listdir(input_file) if f.lower().endswith('.dcm') or f.lower().endswith('.ima')]
        if not input_dicom_files:
            raise ValueError("No DICOM files found in the input folder.")
        
        # Read the first DICOM file to copy relevant attributes
        original_ds = pydicom.dcmread(input_dicom_files[0], stop_before_pixels=True)

        # Create a DICOM series from the SimpleITK image
        dicom_series = sitk.GetArrayFromImage(image)
        num_slices = dicom_series.shape[0]
        
        # Save each slice in the DICOM series
        for i in range(num_slices):
            dicom_slice = original_ds.copy()  # Copy original attributes
            
            dicom_slice.SOPInstanceUID = generate_uid()
            
            # Set pixel data
            pixel_array = dicom_series[i].astype('uint16')
            dicom_slice.PixelData = pixel_array.tobytes()
            
            # Set VR for Pixel Data element to 'OW'
            dicom_slice[0x7FE0, 0x0010].VR = 'OW'
            
            # Set image position and slice thickness attributes
            dicom_slice.ImagePositionPatient = f'0\\0\\{i+1}'
            dicom_slice.SliceThickness = image.GetSpacing()[2]
            
            # Save DICOM file
            output_file = os.path.join(output_folder, f"{i+1:06d}.dcm")
            dicom_slice.save_as(output_file)
        
    else:
       # Create a DICOM series from the SimpleITK image
        dicom_series = sitk.GetArrayFromImage(image)
       
       # Get relevant information from the input image
        image_spacing = image.GetSpacing()
        image_origin = image.GetOrigin()

        # Generate a single StudyInstanceUID for the entire study
        study_uid = generate_uid()
        # Generate a single SeriesInstanceUID for the entire series
        series_uid = generate_uid()

        sop_instance_uids = [generate_uid() for _ in range(dicom_series.shape[0])]

        for i, sop_instance_uid in enumerate(sop_instance_uids):
            dicom = pydicom.Dataset()
            dicom.SOPInstanceUID = sop_instance_uid
            dicom.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4'  # MR Image Storage
            dicom.StudyInstanceUID = study_uid  # Use the same StudyInstanceUID for all images
            dicom.SeriesInstanceUID = series_uid  # Use the same SeriesInstanceUID for all images

            # Set transfer syntax attributes
            dicom.is_little_endian = True
            dicom.is_implicit_VR = False  # Explicit VR Little Endian (default for MR Image Storage)

            # Use information from the input image
            dicom.PixelSpacing = [image_spacing[0], image_spacing[1]]
            dicom.ImagePositionPatient = [image_origin[0], image_origin[1], i * image_spacing[2]]
            dicom.SliceThickness = image_spacing[2]
            dicom.BitsAllocated = 16  # Adjust this based on your image data type (e.g., 8 or 16 bits)

            dicom.RescaleIntercept = 0  # Set the correct RescaleIntercept value based on your data
            dicom.RescaleSlope = 1  # Set the correct RescaleSlope value based on your data

            dicom.SeriesNumber = "100"  # Set the desired SeriesNumber
            dicom.SeriesDescription = "AutoRef Normalized"  # Set the desired SeriesDescription

            dicom.Rows, dicom.Columns = dicom_series.shape[1:]

            # Adjust pixel values using rescale slope and intercept
            pixel_array = dicom_series[i] * dicom.RescaleSlope + dicom.RescaleIntercept

            dicom.PixelData = pixel_array.astype(np.int16).tobytes()
            dicom_file = os.path.join(output_folder, f"{i+1:06d}.dcm")
            dicom.save_as(dicom_file)


def save_image(image, input_file_path, is_dicom, output_file_path):
    """
    Saves the given SimpleITK image object to a file with supported SimpleITK format.
    If the output_file_path has no file extension, it performs sitk_image_to_dicom_series
    to save the image as a DICOM series. Otherwise, it uses sitk.WriteImage to save the image.

    Parameters:
        image (SimpleITK.Image): The image to be saved.
        input_file_path (str): The path to the file/folder(for DICOM).
        is_dicom (bool): Is the input a DICOM folder. 
        output_file_path (str): The path to save the output file.

    Returns:
        None.
    """
    # Check if the output_file_path has no file extension
    if not os.path.splitext(output_file_path)[1]:  
        # Assume it is a folder path and save the image as a DICOM series
        sitk_image_to_dicom_series(image, input_file_path, output_file_path, is_dicom)
    else:
        # Save the image using sitk.WriteImage
        sitk.WriteImage(image, output_file_path)


def perform_n4_bias_field_correction(image, num_iterations=10, convergence_threshold=0.001):
    """
    Performs N4 bias field correction on the input image.

    Parameters:
        image (SimpleITK.Image): The input image to be corrected.
        num_iterations (int): The maximum number of iterations for the N4 bias field correction.
        convergence_threshold (float): The convergence threshold for N4 bias field correction.

    Returns:
        upsampled_corrected_image (SimpleITK.Image): The N4 bias field corrected image.
    """
    # Downsample the image to a lower resolution (optional)
    new_spacing = [2.0, 2.0, 2.0]  # Define the new spacing for downsampling
    downsampled_image = sitk.Image(
        sitk.Resample(image, interpolator=sitk.sitkLinear))
    downsampled_image.SetSpacing(new_spacing)

    # Perform N4 bias field correction on the downsampled image in parallel
    num_threads = multiprocessing.cpu_count()
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetNumberOfThreads(num_threads)
    corrector.SetMaximumNumberOfIterations([num_iterations])  # Pass as a list
    corrector.SetConvergenceThreshold(convergence_threshold)
    corrected_image = corrector.Execute(downsampled_image)

    # Upsample the last corrected image back to the original resolution
    upsampled_corrected_image = sitk.Image(sitk.Resample(
        corrected_image, interpolator=sitk.sitkLinear))
    upsampled_corrected_image.SetSpacing(image.GetSpacing())

    return upsampled_corrected_image


def rescale_image(image, scaling_method, scaling_method_args=None):
    """
    Rescale the input image using the specified scaling method.

    Parameters:
        image (SimpleITK.Image): The input image to be rescaled.
        scaling_method (str): The chosen scaling method ('none', 'max', 'median', or 'percentile').
        scaling_method_args (list or tuple, optional): Additional arguments for the scaling method.

    Returns:
        image (SimpleITK.Image): The rescaled image.
    """
    if scaling_method == 'none':
        # No scaling is applied.
        return image

    if scaling_method == 'max':
        # Scaling by the maximum value multiplied by scaleFactor.
        scaleFactor = scaling_method_args[0]
        return image / (np.max(image) * scaleFactor)

    if scaling_method == 'median':
        # Scaling by the median value multiplied by scaleFactor.
        scaleFactor = scaling_method_args[0]
        return image / (np.median(image) * scaleFactor)

    if scaling_method == 'percentile':
        # Scaling by the given percentile multiplied by scaleFactor.
        prct = scaling_method_args[0]
        scaleFactor = scaling_method_args[1]
        return image / (np.percentile(image, prct) * scaleFactor)

    raise ValueError(
        "Invalid scaling_method. Choose from 'none', 'max', 'median', or 'percentile'.")
    """
    # Example usage:
    # Assuming 'corrected_image' is the output of the resizing step.
    # Replace this with the correct image variable name if needed.


    # Example 1: No scaling
    rescaled_image_none = rescale_image(corrected_image, 'none')

    # Example 2: Scaling by the maximum value multiplied by a scaleFactor (e.g., 2.0)
    rescaled_image_max = rescale_image(
        corrected_image, 'max', scaling_method_args=[2.0])

    # Example 3: Scaling by the median value multiplied by a scaleFactor (e.g., 0.5)
    rescaled_image_median = rescale_image(
        corrected_image, 'median', scaling_method_args=[0.5])

    # Example 4: Scaling by the 95th percentile value multiplied by a scaleFactor (e.g., 0.8)
    rescaled_image_percentile = rescale_image(
        corrected_image, 'percentile', scaling_method_args=[95, 0.8])
        """


def resize_image(image, new_size=(384, 384), new_spacing=(0.5, 0.5), original_size=None, original_spacing=None):
    """
    Resize the input image to the specified new size and spacing, or restore it to its original size.

    Parameters:
        image (SimpleITK.Image): The input image to be resized.
        new_size (tuple, optional): The new size of the image in (rows, cols). Default is (384, 384).
        new_spacing (tuple, optional): The new pixel spacing in (rows, cols) in mm. Default is (0.5, 0.5).
        original_size (tuple, optional): The original size of the image in (rows, cols).
        original_spacing (tuple, optional): The original pixel spacing in (rows, cols) in mm.

    Returns:
        resized_image (SimpleITK.Image): The resized image.
    """
    if original_size is None:
        original_size = image.GetSize()
    if original_spacing is None:
        original_spacing = image.GetSpacing()

    # Set the new size and new spacing
    new_size = new_size + (image.GetSize()[2],)
    new_spacing = new_spacing + (original_spacing[2],)

    # Calculate the new dimensions and padding
    resize_factors = [nsz/osz for nsz, osz in zip(new_size, original_size)]
    new_spacing = [ospc/rf for ospc,
                   rf in zip(original_spacing, resize_factors)]
    new_size_rounded = [int(round(osz*rf))
                        for osz, rf in zip(original_size, resize_factors)]
    pad_required = [nsz-rsz for nsz, rsz in zip(new_size, new_size_rounded)]
    padding = [int(pad // 2) for pad in pad_required]

    # Perform the resizing and padding
    resized_image = sitk.Resample(image, new_size_rounded, sitk.Transform(
    ), sitk.sitkLinear, image.GetOrigin(), new_spacing, image.GetDirection(), 0.0, image.GetPixelIDValue())
    resized_image = sitk.ConstantPad(resized_image, padding, padding, 0.0)

    return resized_image

    """
    # Example usage:
    # Assuming 'rescaled_image' is the output of the N4 bias field correction step.
    # Replace this with the correct image variable name if needed.

    # Resize the 'rescaled_image' to 384x384 pixels with 0.5x0.5 mm spacing
    resized_image = resize_image(rescaled_image, new_size=(384, 384), new_spacing=(0.5, 0.5))

    # Now, to resize it back to its original size and spacing, assuming original_size and original_spacing are known
    restored_image = resize_image(resized_image, new_size=original_size, new_spacing=original_spacing)
    In this updated function, if original_size and original_spacing are provided, it will
    """


def write_slices_to_disk(image_sequence, output_directory):
    """
    Write all the slices from the input image sequence to disk and return a list of written image paths.

    Parameters:
        image_sequence (SimpleITK.Image): The input image sequence (SimpleITK image).
        output_directory (str): The directory where the slices will be saved.

    Returns:
        None.
    """
    num_slices = image_sequence.GetSize()[2]

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # written_image_paths = []

    for slice_index in range(num_slices):
        # Extract the 2D slice from the 3D image sequence
        slice = image_sequence[:, :, slice_index]

        # Convert the slice to numpy array and rescale the pixel values to [0, 255]
        slice_np = sitk.GetArrayViewFromImage(slice)
        slice_np = (slice_np - np.min(slice_np)) / \
            (np.max(slice_np) - np.min(slice_np)) * 255
        slice_np = slice_np.astype(np.uint8)

        # Create a SimpleITK image from the numpy array
        slice_rescaled = sitk.GetImageFromArray(slice_np)

        # Generate the file path for the slice
        file_name = f'{slice_index:02d}.jpg'
        file_path = os.path.join(output_directory, file_name)

        # Write the slice to disk using SimpleITK's WriteImage
        sitk.WriteImage(slice_rescaled, file_path)


def preprocess_image_for_detection(image_path, class_name=None):
    """
    Pre-process an image for input to the YOLOv8 object detection model.
    Here it convert input image to tensor.

    Parameters:
        image_path (str): The path to the input image file.
        class_name (str): The class name to determine the region of interest (ROI).
                          If None, no cropping is applied.

    Returns:
        input (np.ndarray): Numpy array in a shape (3, width, height) where 3 is the number of color channels.
        image_width (int): The width of the original image.
        image_height (int): The height of the original image.
    """
    image = Image.open(image_path)
    image_width, image_height = image.size

    # Crop the image based on the class_name (if specified)
    if class_name == "fat":
        # Crop the lower 50% of the image
        image = image.crop((0, image_height // 2, image_width, image_height))
    elif class_name == "muscle":
        # Crop the upper 50% of the image
        image = image.crop((0, 0, image_width, image_height // 2))

    # Resize the image to the desired input size
    image = image.resize((384, 384))
    image = image.convert("RGB")

    # Convert the image to a numpy array and normalize it
    input = np.array(image) / 255.0

    # Transpose the array to have the channel dimension as the first dimension
    input = input.transpose(2, 0, 1)

    # Reshape the array to have a batch dimension of 1
    input = input.reshape(1, 3, 384, 384)
    input = input.astype(np.float32)

    return input, image_width, image_height


def run_model(input, model_path):
    """
    Predict an image using the YOLOv8 object detection model.
    It pass the provided input tensor to YOLOv8 neural network and return result.

    Parameters:
        input (np.ndarray): Numpy array in a shape (3, width, height).
        model_path (str): The path to the YOLO v8 ONNX model file.

    Returns:
        output (np.ndarray): Raw output of YOLOv8 network as an array.
    """
    # Load the YOLOv8 ONNX model using onnxruntime.
    model = ort.InferenceSession(model_path)

    # Run the inference by passing the input tensor to the model and storing the outputs.
    outputs = model.run(["output0"], {"images": input})

    # Extract the output from the outputs list and return it.
    output = outputs[0]
    return output


def process_model_output(output, image_width, image_height, yolo_classes=["fat", "muscle"],
                         confidence_threshold=0.5, iou_threshold=0.5):
    """
    Process the output from YOLOv8.
    It converts RAW output from YOLOv8 to an array
    of detected objects. Each object contain the bounding box of
    this object, the type of object and the probability

    Parameters:
        output (np.ndarray): Raw output of YOLOv8 network as an array.
        image_width (int): The width of the original image.
        image_height (int): The height of the original image.
        yolo_classes (list): The classes of the trained model.
        confidence_threshold (float): The confidence threshold to filter out low-confidence predictions.
        iou_threshold (float): The intersection over union (IOU) threshold for non-maximum suppression.

    Returns:
        result (list): Array of detected objects in a format [[x1,y1,x2,y2,object_type,probability],..]
    """
    # Convert the raw output to a floating-point array.
    output = output[0].astype(float)

    # Transpose the output array.
    output = output.transpose()

    # Process each row in the output to extract detected objects.
    bboxes = []
    for row in output:
        # Get the maximum probability and the corresponding class ID for the row.
        probability = row[4:].max()
        if probability < confidence_threshold:
            continue
        class_id = row[4:].argmax()

        # Get the label (object type) from the class ID.
        label = yolo_classes[class_id]

        #  Extract the bounding box coordinates (x1, y1, x2, y2) from the row.
        xc, yc, w, h = row[:4]
        x1 = (xc - w/2) / 384 * image_width
        y1 = (yc - h/2) / 384 * image_height
        x2 = (xc + w/2) / 384 * image_width
        y2 = (yc + h/2) / 384 * image_height

        # Append the detected object's information to the boxes list.
        bboxes.append([x1, y1, x2, y2, label, probability])

    # Sort the detected objects by their probabilities in descending order.
    bboxes.sort(key=lambda x: x[5], reverse=True)

    # Perform non-maximum suppression to remove overlapping boxes.
    result = []
    while len(bboxes) > 0:
        result.append(bboxes[0])
        # Filter out boxes with a high intersection over union (IOU) with the currently selected box.
        bboxes = [bbox for bbox in bboxes if iou(
            bbox, bboxes[0]) < iou_threshold]

    # Return the final list of detected objects.
    return result


def iou(box1, box2):
    """
    Calculate the "Intersection-over-union" coefficient for the specified two bounding boxes.

    Parameters:
        box1: First box in format: [x1, y1, x2, y2, object_class, probability].
        box2: Second box in format: [x1, y1, x2, y2, object_class, probability].

    Returns:
        iou_coefficient (float): Intersection over union ratio as a floating-point number.
    """
    # Calculate the intersection area between the two boxes.
    intersection_area = intersection(box1, box2)

    # Calculate the union area between the two boxes.
    union_area = union(box1, box2)

    # Calculate the "Intersection-over-union" coefficient as the ratio of intersection to union.
    iou_coefficient = intersection_area / union_area

    # Return the calculated IOU coefficient as a floating-point number.
    return iou_coefficient


def union(box1, box2):
    """
    Calculate the union area of two bounding boxes.

    Parameters:
        box1: First box in format: [x1, y1, x2, y2, object_class, probability].
        box2: Second box in format: [x1, y1, x2, y2, object_class, probability].

    Returns:
        union_area (float): Area of the boxes union as a floating-point number.
    """
    # Extract the coordinates (x1, y1, x2, y2) of both boxes.
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]

    # Calculate the area of each box.
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    # Calculate the union area by adding the areas of both boxes and subtracting the intersection area.
    union_area = box1_area + box2_area - intersection(box1, box2)

    # Return the union area as a floating-point number.
    return union_area


def intersection(box1, box2):
    """
    Calculate the intersection area of two bounding boxes.

    Parameters:
        box1: First box in format: [x1, y1, x2, y2, object_class, probability].
        box2: Second box in format: [x1, y1, x2, y2, object_class, probability].

    Returns:
        intersection_area (float): Area of intersection of the boxes as a floating-point number.
    """
    # Extract the coordinates (x1, y1, x2, y2) of both boxes.
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]

    # Calculate the coordinates of the intersection area.
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)

    # Calculate the width and height of the intersection area.
    width = x2 - x1
    height = y2 - y1

    # Check if there's an actual intersection by ensuring width and height are non-negative.
    if width <= 0 or height <= 0:
        intersection_area = 0.0
    else:
        # Calculate the intersection area by multiplying width and height.
        intersection_area = width * height

    # Return the intersection area as a floating-point number.
    return intersection_area


def detect_objects_on_image(image_path, model_path):
    """
    Detect the objects in the input image.
    It receives an image, passes it through the YOLOv8 neural network,
    and returns an array of detected objects and their bounding boxes.

    Parameters:
        image_path (str): The path to the input image file.
        model_path (str): The path to the YOLO v8 ONNX model file.

    Returns:
        result (list): Array of bounding boxes in the format [[x1, y1, x2, y2, object_type, probability], ...]
    """
    # Preprocess the input image for detection to get the input tensor and image dimensions.
    input, image_width, image_height = preprocess_image_for_detection(
        image_path)

    # Run the YOLOv8 neural network on the preprocessed input to get the raw output.
    output = run_model(input, model_path)

    # Process the YOLOv8 model's output to get the final list of detected objects.
    result = process_model_output(output, image_width, image_height)

    # Return the list of detected objects and their bounding boxes.
    return result


def plot_images_with_bboxes(image_folder, top_predictions, output_folder):
    """
    Plot the images with bounding boxes overlaid on them.

    Parameters:
        image_folder (str): The path to the folder containing the input images.
        top_predictions (dict): A dictionary containing the top 3 predictions for each class.
                                The keys are class names, and the values are lists of dictionaries.
                                Each dictionary contains the 'slice' name (without the ".jpg"),
                                the bounding box coordinates, and the probability score.
        output_folder (str): The path to the folder where the plotted images will be saved.
    """
    for class_name, predictions in top_predictions.items():
        for prediction in predictions:
            image_name = prediction['slice'] + '.jpg'
            image_path = os.path.join(image_folder, image_name)

            # Read the image
            image = plt.imread(image_path)

            # Create figure and axes
            fig, ax = plt.subplots()

            # Display the image
            ax.imshow(image)

            # Get bounding box coordinates
            x1, y1, x2, y2 = prediction['bbox']

            # Calculate box width and height
            box_width = x2 - x1
            box_height = y2 - y1

            # Create a Rectangle patch
            rect = patches.Rectangle(
                (x1, y1), box_width, box_height, linewidth=2, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

            # Add a text label with the class name and probability
            ax.text(
                x1, y1, f"{class_name} - {prediction['probability']:.2f}", fontsize=12, color='r')

            # Set axis limits to match the image size
            ax.set_xlim(0, image.shape[1])
            ax.set_ylim(image.shape[0], 0)

            # Remove axis ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            # Save the plot to the output folder with the same image name
            output_path = os.path.join(
                output_folder, f"{prediction['slice']}_{class_name}.png")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

    """
    # Example usage:
    # Plot bbox on image
    # output_folder = r"C:\Data\Postdoc\pyAutoRef\bbox-Plots"
    # plot_images_with_bboxes(input_folder, top_predictions, output_folder)
    """


def detect_remove_outliers(image_3D):
    """
    Detect and remove outliers in a 3D image using the IQR method.

    Parameters:
        image_3D (SimpleITK.Image): The 3D image.

    Returns:
        image_3D_wo_outliers (SimpleITK.Image): The 3D image with outliers removed.
    """
    # Convert the SimpleITK image to a NumPy array.
    image_array = sitk.GetArrayFromImage(image_3D)

    # Calculate the Interquartile Range (IQR) to detect outliers.
    iqr = np.percentile(image_array, 75) - np.percentile(image_array, 25)

    # Create a copy of the image array to preserve the original data.
    image_array_wo_outliers = image_array.copy()

    # Set outlier values (above Q3 + 3*IQR) to NaN to remove them.
    image_array_wo_outliers[image_array > np.percentile(
        image_array, 75) + 3 * iqr] = np.nan

    # Convert the processed NumPy array back to a SimpleITK image.
    image_3D_wo_outliers = sitk.GetImageFromArray(image_array_wo_outliers)

    # Copy the metadata (information) from the original image to the processed image.
    image_3D_wo_outliers.CopyInformation(image_3D)

    # Return the 3D image with outliers removed.
    return image_3D_wo_outliers


def extract_largest_connected_component(binary_image):
    """
    Extract the largest connected component (object) from the binary image.

    Parameters:
        binary_image (numpy.ndarray): The binary image.

    Returns:
        largest_component (numpy.ndarray): The binary image containing only the largest connected component.
    """
    # Label connected regions in the binary image using 8-connectivity.
    labeled_image, num_labels = measure.label(
        binary_image, connectivity=2, return_num=True)

    # Count the occurrences of each label in the labeled image.
    unique_labels, label_counts = np.unique(labeled_image, return_counts=True)

    # Find the label corresponding to the largest connected component (excluding the background label).
    largest_label = unique_labels[np.argmax(label_counts[1:]) + 1]

    # Create a binary image containing only the largest connected component.
    largest_component = labeled_image == largest_label

    # Return the binary image containing only the largest connected component.
    return largest_component


def extract_intensities_from_mask(image_slice, largest_component_mask):
    """
    Extract the intensities from the image slice corresponding to the masked region.

    Parameters:
        image_slice (numpy.ndarray): The image slice.
        largest_component_mask (numpy.ndarray): The binary mask of the largest connected component.

    Returns:
        intensities (numpy.ndarray): The intensities of the pixels in the masked region.
    """
    masked_intensities = image_slice[largest_component_mask.astype(bool)]
    return masked_intensities


def overlay_mask_on_image(class_name, slice_number, cropped_image, largest_component, output_folder):
    """
    Overlay the mask on the image slice, save the plot, and return the masked image.

    Parameters:
        class_name (str): The name of the class.
        slice_number (int): The slice number.
        cropped_image (numpy.ndarray): The cropped image slice.
        largest_component (numpy.ndarray): The binary mask of the largest connected component.
        output_folder (str): The folder path to save the plot.

    Returns:
        numpy.ndarray: The masked image with the overlay.

    """
    # Copy the cropped image to work on
    image_with_mask = np.copy(cropped_image)

    # Set masked region to maximum intensity (white)
    image_with_mask[largest_component > 0] = 255

    # Create a figure and plot the image with the overlaid mask
    plt.figure()
    plt.imshow(image_with_mask, cmap='gray')
    plt.title(f"Class: {class_name} - Slice: {slice_number}")
    plt.axis('off')

    # Save the plot to the output folder with a descriptive filename
    output_path = os.path.join(
        output_folder, f"{class_name}_slice{slice_number:03}.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return image_with_mask

    """
    # Example usage:
    
    # In post_process_predictions function in post_processing.py
    image_with_mask = overlay_mask_on_image(class_name, slice_number, cropped_image, largest_component, output_folder)

    # General example
    class_name = "fat"
    slice_number = 10
    cropped_image = np.random.randint(0, 256, size=(256, 256), dtype=np.uint8)
    largest_component = np.random.randint(0, 2, size=(256, 256), dtype=np.uint8)
    output_folder = r"C:\Data\Postdoc\pyAutoRef\roi-output"

    # Overlay the mask on the image slice and save the plot
    image_with_mask = overlay_mask_on_image(class_name, slice_number, cro
     """


def suppress_warnings(func):
    """
    A decorator to suppress warning messages while executing a function.

    Parameters:
        func (function): The function to be decorated.

    Returns:
        function: A new function that suppresses warnings while executing the original function.

    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        The wrapper function that suppresses warnings and executes the original function.

        Parameters:
            *args: Variable-length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The return value of the original function.

        """
        # Create a logger to handle the warnings
        logger = logging.getLogger(func.__name__)
        logger.setLevel(logging.ERROR)  # Set the level to ERROR to suppress all warnings

        with warnings.catch_warnings():
            # Redirect the warnings to the custom logger
            warnings.simplefilter("always")
            warnings.showwarning = lambda *args, **kwargs: logger.warning(*args, **kwargs)
            return func(*args, **kwargs)
    return wrapper

    """
        Example usage:
        @suppress_warnings
        def your_function():
            # Your function code here
            # ...

        # Call the function, and warnings will be suppressed
        your_function()
    """