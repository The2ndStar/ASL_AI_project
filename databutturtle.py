import cv2
import csv
import time
import os
import numpy as np
import subprocess
import sys
from handtracking import HandDetector

import pygame  


pygame.mixer.init()
pygame.mixer.music.load(r"sound\bgmusic.mp3") 
pygame.mixer.music.play(-1)  


alphabet_img = {
    'W': cv2.imread("image/W.jpg"),
    'H': cv2.imread("image/H.jpg"),
    'C': cv2.imread("image/C.jpg"),
    'D': cv2.imread("image/D.jpg"),
    'Y': cv2.imread("image/Y.jpg")
}

for gest, img in alphabet_img.items():
    if img is None:
        print(f"Warning: Failed to load image for gesture {gest}. Please check the file path.")

def collect_gest():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1, complexity=0)

    gestures = ['W', 'H', 'C', 'D', 'Y']
    samples_per_gesture = 100

    folder = "gamejoint"
    if not os.path.exists(folder):
        os.makedirs(folder)

    for gest in gestures:
        print(f"Collecting data for gesture: {gest}")
        collecting = False

        while True:
            success, img = cap.read()
            if not success:
                print("Failed to capture image.")
                break

            gest_img = alphabet_img.get(gest)
            if gest_img is not None:
                gesture_img_resized = cv2.resize(gest_img, (200, 200))
                img[50:50 + gesture_img_resized.shape[0], 50:50 + gesture_img_resized.shape[1]] = gesture_img_resized
            else:
                cv2.putText(img, f"No image for {gest}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.putText(img, f"Perform the {gest} gesture", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, "Press 'S' to start", (50, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Data Collection", img)

            key = cv2.waitKey(1)
            if key == ord("s"):
                collecting = True
                break
            elif key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return

        if not collecting:
            continue

        csv_filename = os.path.join(folder, f"{gest}_joints.csv")
        with open(csv_filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label"] + [f"{i}_x" for i in range(21)] + [f"{i}_y" for i in range(21)])

        for sample in range(samples_per_gesture):
            print(f"Sample {sample + 1}/{samples_per_gesture} for {gest}...")

            success, img = cap.read()
            if not success:
                print("Failed to capture image.")
                break

            img, bboxes = detector.findHands(img)
            lmList = []

            if bboxes:
                lmList = detector.findPosition(img, draw=False)

                if lmList:
                    joint_data = [sample + 1, gest]
                    for joint in lmList:
                        joint_data.extend(joint[1:3])

                    with open(csv_filename, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(joint_data)

            cv2.putText(img, f"Gesture: {gest}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, f"Sample: {sample + 1}/{samples_per_gesture}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Data Collection", img)

            if cv2.waitKey(1) == ord("q"):
                break
    
    pygame.mixer.music.stop() 
    cap.release()
    cv2.destroyAllWindows()


    print("Returning to main menu...")
    subprocess.run([sys.executable, "start.py"])


if __name__ == "__main__":
    collect_gest()