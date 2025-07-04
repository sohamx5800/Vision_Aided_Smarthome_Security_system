# AI Door Lock System

## Project Overview
This repository contains the full implementation and documentation of the AI Door Lock System, a sophisticated security solution developed as my final year engineering project. Leveraging Python, OpenCV for face detection and recognition, `speechrecognition` for voice commands, `pyttsx3` for text-to-speech feedback, and serial communication with an Arduino, this system replaces traditional keys with biometric face recognition. The project captures user faces, trains a machine learning model, and stores data in a structured dataset, enabling seamless door unlocking for authorized individuals. All code, configurations, diagrams, and instructions are included, making it an educational and practical showcase of AI in real-world applications.

## Detailed Workflow
The system operates through a well-defined sequence of steps to ensure robust face recognition and door control:
1. **System Initialization**: 
   - Launches with `cv2.VideoCapture` to initialize the webcam and establishes a serial connection to an Arduino on COM7.
   - Loads configuration settings (e.g., servo angles, lock delay) from `config.json`.
2. **Face Detection**:
   - Continuously processes the webcam feed using `haarcascade_frontalface_default.xml` to detect faces in real-time.
   - Displays the feed in a "Webcam" window with detected face rectangles.
3. **Handling Unknown Faces**:
   - Upon detecting an unfamiliar face, the system announces "Hey bro, Someone is outside the door. Should I open the door?" via `pyttsx3`.
   - Listens for a response using `speechrecognition` to determine the next action.
4. **Image Capture Process**:
   - If the user says "remember," it initiates a guided capture sequence:
     - Prompts "Please place your face front to capture," waits 2 seconds, and captures 2 images.
     - Repeats for "right side," "left side," "top tilt," and "bottom tilt," collecting 2 images per angle (total 10 images).
   - Uses `face_cascade.detectMultiScale` to extract face regions, stored in `face_samples` with a unique `current_id` in `ids`.
5. **Training and Data Storage**:
   - After capturing 10 images, announces "Image capturing successfully."
   - Prompts "Please speak your name" and extracts the name (e.g., "I am Soham" → "soham") using `extract_name_from_response`.
   - Trains an LBPH (Local Binary Patterns Histograms) model with `recognizer_model.train(face_samples, np.array(ids))`.
   - Creates a folder (e.g., "Home Mem Datasets/soham") and saves the model as "soham_model.yml" using `recognizer_model.save`.
   - Updates the `known_faces` dictionary for future recognition.
6. **Face Recognition and Door Control**:
   - Matches real-time face images against trained models using `recognizer_model.predict()`, with a confidence threshold <80.
   - If recognized (e.g., "soham"), announces "soham is outside the door. I am opening the door" and sends 'O' to the Arduino to unlock.
   - Locks the door after a configurable delay (from `config.json`) and confirms with "The door is now locked again."
7. **Exit**: Press 'q' to terminate the program, releasing resources.

## Features
- **Real-Time Face Detection**: Utilizes OpenCV’s Haar Cascade Classifier for accurate face detection.
- **Multi-Angle Capture**: Collects 10 images across 5 angles for robust training data.
- **LBPH Recognition**: Employs Local Binary Patterns Histograms for effective face recognition.
- **Voice Interaction**: Provides voice prompts and feedback for user-friendly operation.
- **Dataset Management**: Organizes trained models in a hierarchical folder structure.
- **Arduino Integration**: Controls a physical door lock via serial commands.
- **Configurable Settings**: Allows customization via `config.json` (e.g., camera index, lock delay).

Complete project detail report: 
### Workflow Diagram[Vision Aided system.pdf](https://github.com/user-attachments/files/21059098/Vision.Aided.system.pdf)

![Workflow Flowchart]([AI Door Lock Flowchart_page-0001](https://github.com/user-attachments/assets/5fb3b6cb-22ff-4367-bc09-c9a448022291))
- Depicts the end-to-end process from face detection to door unlocking.

### Dataset Structure
![Dataset Flowchart](![data train and gather  workflow_page-0001](https://github.com/user-attachments/assets/26ba025c-c6c0-451e-ad42-7a29cbcf0df2))
- Illustrates the "Home Mem Datasets" folder with user subfolders and `.yml` files.

## Installation
1. **Clone the Repository**:git clone https://github.com/sohamx5800/Vision_Aided_Smarthome_Security_system.git
. **Install Dependencies**: pip install -r requirements.txt
   
      
. **Hardware Setup**:
- Connect a webcam for face detection.
- Attach an Arduino to COM7 and configure it with the corresponding servo code.
- Ensure `config.json` matches your hardware setup.
  
 2. **Initial Detection**:
- Position yourself in front of the webcam. For an unknown face, the system will prompt for action.
3. **Register a New User**:
- Say "remember," follow voice prompts to adjust your face angles, and speak your name (e.g., "I am Soham").
- Wait for "Image capturing successfully" and confirmation "Thank you, soham. I will remember you."
4. **Access the Door**:
- The system will recognize you on subsequent visits, unlocking the door automatically.
5. **Exit**:
- Press 'q' to close the program and release the camera and serial port.

## Repository Contents
- `Ai_room_lock.py`: The core Python script implementing the door lock logic.
- `flowchart_workflow.png`: Detailed flowchart of the operational workflow.
- `flowchart_dataset.png`: Diagram of the dataset organization.
- `haarcascade_frontalface_default.xml`: Pre-trained classifier for face detection.
- `config.json`: JSON file for configurable parameters (e.g., camera index, servo angles).
- `README.md`: This comprehensive documentation file.

## Future Improvements
- **Performance Optimization**: Reduce capture time by streamlining angle adjustments.
- **Accuracy Enhancement**: Integrate deep learning models (e.g., DeepFace) for better recognition.
- **User Interface**: Develop a GUI using Tkinter or PyQt for enhanced user interaction.
- **Remote Access**: Add network capabilities for remote door control.

## Contact
- **Author**: [Soham Adhikary]  
- **Email**: [sohamadhikary707@gmail.com]  
- **GitHub**: [https://github.com/sohamx5800 ]  
- **LinkedIn**: [https://www.linkedin.com/in/soham-adhikary-614511221/]     
