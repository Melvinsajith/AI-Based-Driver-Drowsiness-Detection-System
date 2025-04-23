# AI-Based Driver Drowsiness Detection System

This project implements a real-time driver monitoring system that detects drowsiness and yawning using facial landmarks. It uses computer vision and machine learning techniques with the help of `dlib`, `OpenCV`, and `pygame` to alert the driver through an audio alarm when signs of fatigue or drowsiness are detected.

## 🚀 Features

- Real-time webcam-based facial monitoring
- Eye Aspect Ratio (EAR) for drowsiness detection
- Mouth Aspect Ratio (MAR) for yawn detection
- Audio alert using `pygame` when drowsiness or yawning is detected
- Visual feedback through bounding contours and EAR/MAR values on the video feed

## 🧠 Technologies Used

- Python 3.x
- OpenCV
- dlib
- imutils
- scipy
- pygame


## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/melvinsajith/AI-Based-Driver-Drowsiness-Detection-System.git
   cd AI-Based-Driver-Drowsiness-Detection-System


2 **Install dependencies**

pip install -r requirements.txt

3 **Download the model file**

    Download the shape_predictor_68_face_landmarks.dat file from dlib's official link
    or https://drive.google.com/file/d/1PLtLdNgPFxjhNRYL-v2OwXXeJj1HZNEs/view?usp=sharing

    Extract and place it in the models/ folder

4. **Run the application**

python app.py

5. **Quit the app**

    Press q while the webcam window is open


## Notes

    Make sure your webcam is working and accessible by OpenCV.

    The audio alert file (alert.wav) must be placed in the specified audio/ directory.

    The script will draw contours around eyes and mouth and print status ("Drowsy", "Yawning", or "Normal") in the terminal.

