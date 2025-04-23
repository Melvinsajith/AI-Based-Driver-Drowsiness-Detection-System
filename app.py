from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import pygame

# Initialize audio
pygame.mixer.init()
pygame.mixer.music.load('/media/melvin/ALTROZ/MELVIN/aiproject/Drowsiness/Drowsiness/audio/alert.wav')

# Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[13], mouth[19])  # 14-20
    B = distance.euclidean(mouth[14], mouth[18])  # 15-19
    C = distance.euclidean(mouth[15], mouth[17])  # 16-18
    D = distance.euclidean(mouth[12], mouth[16])  # 13-17 (horizontal)
    return (A + B + C) / (3.0 * D)

# Thresholds
thresh_drowsiness = 0.25
thresh_yawn = 0.6
frame_check = 20

# Load models
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("/media/melvin/ALTROZ/MELVIN/aiproject/Drowsiness/Drowsiness/models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

cap = cv2.VideoCapture(0)
flag_drowsy = 0
flag_yawn = 0
alarm_active = False

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        # EAR & MAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # Draw eyes and mouth
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (255, 0, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (255, 0, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)

        status = "Normal"
        color = (0, 255, 0)

        # Drowsiness Detection
        if ear < thresh_drowsiness:
            flag_drowsy += 1
            if flag_drowsy >= frame_check:
                if not alarm_active:
                    pygame.mixer.music.play(-1)
                    alarm_active = True
                status = "Drowsy"
                color = (0, 0, 255)
                print("Drowsy")
        else:
            flag_drowsy = 0

        # Yawning Detection
        if mar > thresh_yawn:
            flag_yawn += 1
            if flag_yawn >= frame_check:
                if not alarm_active:
                    pygame.mixer.music.play(-1)
                    alarm_active = True
                status = "Yawning"
                color = (0, 0, 255)
                print("Yawning")
        else:
            flag_yawn = 0

        # Reset alarm if no condition met
        if flag_drowsy == 0 and flag_yawn == 0:
            if alarm_active:
                pygame.mixer.music.stop()
                alarm_active = False
            print("Normal")

        # Overlay EAR, MAR, and status
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Status: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
