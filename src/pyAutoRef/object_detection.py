import os

from pyAutoRef.utils import detect_objects_on_image


def object_detection(input_folder, model_path, yolo_classes=["fat", "muscle"], slice_percent=[0.15, 0.85]):
    """
    Perform prediction on multiple images in an input folder and select the top 3 images
    with the highest prediction score for each class.

    Parameters:
        input_folder (str): The path to the folder containing the input images.
        model_path (str): The path to the YOLO v8 ONNX model file.
        yolo_classes (list): The classes of the trained model. Default is ["fat", "muscle"].
        slice_percent (list): The percentage range of slices to be detected. Default is [0.15, 0.85]. Remove the first 15% and the last 15% of slices.

    Returns:
        top_predictions (dict): A dictionary containing the top 3 predictions for each class.
                                The keys are class names, and the values are lists of dictionaries.
                                Each dictionary contains the 'slice' name (without the ".jpg"),
                                the bounding box coordinates, and the probability score.
    """
    # Initialize a dictionary to store the top 3 predictions for each class.
    top_predictions = {class_name: [] for class_name in yolo_classes}

    # Get a list of image filenames in the input folder.
    image_filenames = [filename for filename in os.listdir(
        input_folder) if filename.endswith(".jpg")]

    # Sort the image filenames (assuming filenames are in the format '00.jpg', '01.jpg', etc.)
    image_filenames.sort()

    # Calculate the number of images
    num_images = len(image_filenames)

    # Calculate the middle part range
    start_index = int(num_images * slice_percent[0])
    end_index = int(num_images * slice_percent[1])

    # Loop through the images in the input folder and perform object detection on them.
    for filename in image_filenames[start_index:end_index]:
        image_path = os.path.join(input_folder, filename)
        result = detect_objects_on_image(image_path, model_path)

        # For each detected object, store the information in the top_predictions dictionary.
        for prediction in result:
            class_name = prediction[4]
            top_predictions[class_name].append({
                'slice': os.path.splitext(filename)[0],
                'bbox': prediction[:4],
                'probability': prediction[5]
            })

    # Sort the predictions by probability score in descending order for each class.
    for class_name in yolo_classes:
        top_predictions[class_name].sort(
            key=lambda x: x['probability'], reverse=True)
        top_predictions[class_name] = top_predictions[class_name][:3]

    # Return the final dictionary containing the top 3 predictions for each class.
    return top_predictions
