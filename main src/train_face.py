import cv2
import os
import numpy as np

def train_face(image_path, person_name, dataset_folder="Home Mem Datasets"):
    """
    Process the captured image and save the trained data as a matrix in a YAML file.
    Args:
        image_path (str): Path to the captured image.
        person_name (str): Name of the person.
        dataset_folder (str): Folder where the data is stored.
    Returns:
        bool: True if training and saving were successful, False otherwise.
    """
    try:
        # Read the captured image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return False

        # Normalize the image to improve matching accuracy
        image = cv2.equalizeHist(image)  # Histogram equalization for better contrast
        image = cv2.resize(image, (100, 100))  # Resize to a standard size for consistency

        # Convert the image to a numpy array (matrix)
        image_matrix = np.array(image, dtype=np.uint8)

        # Load existing YAML data (if any)
        yaml_path = os.path.join(dataset_folder, "faces.yaml")
        fs = cv2.FileStorage(yaml_path, cv2.FileStorage_READ)

        # If the file doesn't exist or is empty, create a new one
        if not fs.isOpened():
            fs.release()
            fs = cv2.FileStorage(yaml_path, cv2.FileStorage_WRITE)
        else:
            # Read existing data to append
            fs.release()
            fs = cv2.FileStorage(yaml_path, cv2.FileStorage_APPEND)

        # Write the matrix to the YAML file under the person's name
        fs.write(f"face_{person_name}", image_matrix)
        fs.release()

        print(f"Trained data for {person_name} saved in {yaml_path}")
        return True
    except Exception as e:
        print(f"Error during training: {e}")
        return False

if __name__ == "__main__":
    # Example usage (for testing)
    train_face("Home Mem Datasets/test_person/test_person.jpg", "test_person")