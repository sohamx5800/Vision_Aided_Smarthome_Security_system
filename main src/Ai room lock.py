import os
import cv2
import pyttsx3
import serial
import time
import speech_recognition as sr
import json
import numpy as np

from train_face import train_face  # Assuming this is for compatibility; we'll replace its usage

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 180)

CONFIG_FILE = "config.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {"servo1_angle": 180, "servo2_angle": 180, "unlock_angle1": 45, "unlock_angle2": 45, "lock_delay": 7, "camera_index": 1}

config = load_config()

# Initialize camera with retry mechanism
def initialize_camera(camera_index):
    for attempt in range(3):  # Retry up to 3 times
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Use DirectShow backend for better compatibility on Windows
        if cap.isOpened():
            print(f"Camera opened successfully on index {camera_index}.")
            return cap
        print(f"Failed to open camera on index {camera_index}, attempt {attempt + 1}/3.")
        cap.release()
        time.sleep(1)  # Wait before retrying
    print(f"Error: Could not open camera on index {camera_index} after 3 attempts.")
    exit()

cap = initialize_camera(config["camera_index"])

try:
    ser = serial.Serial('COM7', 9600)  # Check your Arduino COM port
    time.sleep(2)
    print("Serial port opened successfully.")
except Exception as e:
    print(f"Error opening serial port: {e}")
    cap.release()
    exit()

# Send configuration data to Arduino
ser.write(json.dumps(config).encode())
time.sleep(3)
print("Sent configuration data to Arduino")

# Face recognition variables
door_unlocked = False
recognizer = sr.Recognizer()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
dataset_folder = "Home Mem Datasets"
known_faces = {}
recognizer_model = cv2.face.LBPHFaceRecognizer_create()  # Use LBPH for robust recognition
trained = False

# Ensure dataset folder exists
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

# Function to recognize speech using Google's speech recognition
def recognize_speech():
    try:
        with sr.Microphone() as source:
            print("Listening for speech...")
            audio = recognizer.listen(source)
            response = recognizer.recognize_google(audio).lower()
            print(f"Recognized speech: {response}")
            return response
    except sr.UnknownValueError:
        print("Speech not understood.")
        engine.say("Sorry, I couldn't understand that.")
        engine.runAndWait()
        return None
    except sr.RequestError as e:
        print(f"Speech recognition error: {e}")
        engine.say("Sorry, there was an issue with the recognition service.")
        engine.runAndWait()
        return None

# Function to extract the actual name from the raw speech input
def extract_name_from_response(raw_name):
    if raw_name:
        if "i am" in raw_name:
            person_name = raw_name.split("i am")[-1].strip()
        elif "my name is" in raw_name:
            person_name = raw_name.split("my name is")[-1].strip()
        else:
            person_name = raw_name.strip()
        person_name = person_name.replace(" ", "_").lower()
        print(f"Extracted name: {person_name}")
        return person_name
    return None

# Function to capture face data (no training or saving here)
def capture_face_data():
    print("Capturing face data...")
    engine.say("Please place your face front to capture.")
    engine.runAndWait()
    time.sleep(2)  # Give user time to adjust

    face_samples = []
    ids = []
    current_id = len(os.listdir(dataset_folder)) if os.path.exists(dataset_folder) else 0

    # Capture 10 images across different angles
    angles = ["front", "right side", "left side", "top tilt", "bottom tilt"]
    for angle in angles:
        print(f"Capturing {angle}...")
        engine.say(f"Please turn your face to the {angle} to capture.")
        engine.runAndWait()
        time.sleep(2)  # Allow user to adjust pose

        count = 0
        while count < 2:  # 2 images per angle for 10 total
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_roi = gray[y:y+h, x:x+w]
                face_samples.append(face_roi)
                ids.append(current_id)
                count += 1
                print(f"Captured {count} images for {angle}.")

            cv2.imshow("Webcam", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):  # 100ms delay
                break

    cv2.imshow("Webcam", frame)  # Keep displaying the last frame
    if len(face_samples) < 10:
        print("Not enough images captured. Capture aborted.")
        engine.say("Not enough images captured. Please try again.")
        engine.runAndWait()
        return None, None

    print("Image capturing successfully.")
    engine.say("Image capturing successfully.")
    engine.runAndWait()
    return face_samples, ids

# Load known faces and trained models
def load_trained_faces():
    global trained
    if os.path.exists(dataset_folder):
        for person_name in os.listdir(dataset_folder):
            person_dir = os.path.join(dataset_folder, person_name)
            if os.path.isdir(person_dir):
                model_path = os.path.join(person_dir, f"{person_name}_model.yml")
                if os.path.exists(model_path):
                    recognizer_model.read(model_path)
                    known_faces[person_name] = person_name  # Store name for mapping
                    trained = True
                    print(f"Loaded trained model for {person_name}")
                else:
                    print(f"No model found for {person_name}")

load_trained_faces()

def send_servo_command(command):
    ser.write((command + '\n').encode())
    print(f"Sent command: {command}")

try:
    while True:
        # Only process frames if the door is not in the process of unlocking/locking
        if not door_unlocked:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # Check for faces in the current frame
            if len(faces) > 0:
                print(f"Face detected: {len(faces)} face(s).")
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    detected_person = None

                    if trained:
                        # Predict using the trained recognizer
                        label, confidence = recognizer_model.predict(face_roi)
                        if confidence < 80:  # Threshold for recognition confidence
                            for name, _ in known_faces.items():
                                if label == list(known_faces.keys()).index(name):
                                    detected_person = name
                                    print(f"Detected Person: {detected_person} with confidence {confidence}")
                                    break
                    else:
                        print("No trained model available.")

                    if detected_person and not door_unlocked:
                        print(f"Speaking: {detected_person} is outside the door. I am opening the door.")
                        engine.say(f"{detected_person} is outside the door. I am opening the door.")
                        engine.runAndWait()
                        try:
                            send_servo_command('O')
                            ser.flush()
                            print("Unlocking successful.")
                            
                            door_unlocked = True
                            time.sleep(config["lock_delay"])
                        
                            print("Speaking: The door is now locked again.")
                            engine.say("The door is now locked again.")
                            engine.runAndWait()
                            
                            door_unlocked = False
                            ser.flushInput()
                            ser.flushOutput()
                            print("Locking successful.")
                        except Exception as e:
                            print(f"Error in sending unlock command: {e}")
                            ser.flush()

                    elif not detected_person and not door_unlocked:
                        print("Speaking: Hey bro, Someone is outside the door. Should I open the door?")
                        engine.say("Hey bro, Someone is outside the door. Should I open the door?")
                        engine.runAndWait()

                        with sr.Microphone() as source:
                            print("Listening for response (Yes or No)...")
                            audio = recognizer.listen(source)

                        try:
                            response = recognizer.recognize_google(audio).lower()
                            print("You said:", response)
                            if 'yes' in response or 'open' in response or 'unlock' in response:
                                print("Unlocking the door...")
                                try:
                                    send_servo_command('O')
                                    ser.flush()
                                    print("Unlocking successful.")
                                    print("Speaking: The door is unlocked, bro.")
                                    engine.say("The door is unlocked, bro.")
                                    engine.runAndWait()
                                    door_unlocked = True
                                    time.sleep(config["lock_delay"])
                                    
                                    print("Speaking: The door is locked again.")
                                    engine.say("The door is locked again.")
                                    engine.runAndWait()
                                    door_unlocked = False
                                    ser.flushInput()
                                    ser.flushOutput()
                                    print("Locking successful.")

                                except Exception as e:
                                    print(f"Error in sending unlock command: {e}")
                                    ser.flush()
                                
                            elif 'no' in response:
                                print("Speaking: The door will remain locked, bro.")
                                engine.say("The door will remain locked, bro.")
                                engine.runAndWait()
                            elif 'remember' in response:
                                print("Speaking: Please place your face front to capture.")
                                engine.say("Please place your face front to capture.")
                                engine.runAndWait()

                                # Capture images first
                                face_samples, ids = capture_face_data()
                                if face_samples is not None and ids is not None:
                                    print("Speaking: Please speak your name.")
                                    engine.say("Please speak your name.")
                                    engine.runAndWait()

                                    raw_name = recognize_speech()
                                    person_name = extract_name_from_response(raw_name)

                                    if person_name:
                                        person_dir = os.path.join(dataset_folder, person_name)
                                        if not os.path.exists(person_dir):
                                            os.makedirs(person_dir)
                                        recognizer_model.train(face_samples, np.array(ids))
                                        recognizer_model.save(os.path.join(person_dir, f"{person_name}_model.yml"))
                                        known_faces[person_name] = person_name
                                        print(f"Trained and saved model for {person_name}")
                                        engine.say(f"Thank you, {person_name}. I will remember you.")
                                        engine.runAndWait()
                                    else:
                                        print("Speaking: Failed to recognize your name. Please try again.")
                                        engine.say("Failed to recognize your name. Please try again.")
                                        engine.runAndWait()

                        except sr.UnknownValueError:
                            print("Speaking: I couldn't understand your response. The door will remain locked.")
                            engine.say("I couldn't understand your response. The door will remain locked.")
                            engine.runAndWait()

        # Always display the camera feed, even when the door is unlocked
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera.")
            break
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    if ser and ser.is_open:
        ser.close()
        print("Serial port closed.")