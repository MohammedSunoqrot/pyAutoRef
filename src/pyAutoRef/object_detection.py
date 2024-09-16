import os
import logging
from pyAutoRef.utils import detect_objects_on_image

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def object_detection(input_folder, model_path, yolo_classes=None, slice_percent=None):
    """
    Perform object detection on multiple images in an input folder and select the top 3 images
    with the highest prediction scores for each class.

    Parameters:
        input_folder (str): The path to the folder containing the input images.
        model_path (str): The path to the YOLO v8 ONNX model file.
        yolo_classes (list, optional): The classes of the trained model. Default is ["fat", "muscle"].
        slice_percent (list, optional): The percentage range of slices to be detected. Default is [0.15, 0.85]. 
                                         Removes the first 15% and the last 15% of slices.

    Returns:
        top_predictions (dict): A dictionary containing the top 3 predictions for each class.
                                The keys are class names, and the values are lists of dictionaries.
                                Each dictionary contains the 'slice' name (without the ".jpg"),
                                the bounding box coordinates, and the probability score.
    """
    if yolo_classes is None:
        yolo_classes = ["fat", "muscle"]
    if slice_percent is None:
        slice_percent = [0.15, 0.85]

    # Initialize a dictionary to store the top 3 predictions for each class.
    top_predictions = {class_name: [] for class_name in yolo_classes}

    # Get a list of image filenames in the input folder.
    try:
        image_filenames = [filename for filename in os.listdir(
            input_folder) if filename.endswith(".jpg")]
    except FileNotFoundError as e:
        logging.error(f"Input folder not found: {e}")
        raise

    # Sort the image filenames
    image_filenames.sort()

    # Calculate the number of images
    num_images = len(image_filenames)

    if num_images == 0:
        logging.warning("No images found in the input folder.")
        return top_predictions

    # Calculate the range of slices to process
    start_index = int(num_images * slice_percent[0])
    end_index = int(num_images * slice_percent[1])

    # Perform object detection on the selected range of images
    for filename in image_filenames[start_index:end_index]:
        image_path = os.path.join(input_folder, filename)
        try:
            result = detect_objects_on_image(image_path, model_path)
        except Exception as e:
            logging.error(f"Error detecting objects in {filename}: {e}")
            continue

        # Store the prediction results
        for prediction in result:
            class_name = prediction[4]
            if class_name in top_predictions:
                top_predictions[class_name].append({
                    'slice': os.path.splitext(filename)[0],
                    'bbox': prediction[:4],
                    'probability': prediction[5]
                })

    # Sort and select top 3 predictions for each class
    for class_name in yolo_classes:
        top_predictions[class_name].sort(
            key=lambda x: x['probability'], reverse=True)
        top_predictions[class_name] = top_predictions[class_name][:3]
        logging.info(
            f"Top predictions for class {class_name}: {top_predictions[class_name]}")

    # Return the final dictionary containing the top 3 predictions for each class.
    return top_predictions
