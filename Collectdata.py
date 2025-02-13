import cv2
import math
import os
import time
import numpy as np
from handtracking import HandDetector  

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, complexity=0)  

offset = 20
imgSize = 300
counter = 0
save_interval = 0.5
last_save = time.time()

folder = "data/B"
if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    if not success:
        break

    img, bboxes = detector.findHands(img)
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  

    if bboxes:
        x, y, w, h = bboxes['bbox']

        x, y, w, h = max(0, x - offset), max(0, y - offset), w + 2 * offset, h + 2 * offset
        imgCrop = img[y:y + h, x:x + w]

        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            C = imgSize / h
            HC = math.ceil(C * w)
            imgReSize = cv2.resize(imgCrop, (HC, imgSize))
            imgReSizeHeight, imgReSizeWidth = imgReSize.shape[:2]
            hGap = math.ceil((imgSize - HC) / 2)
            imgWhite[0:imgReSizeHeight, hGap:HC + hGap] = imgReSize
        else:
            C = imgSize / w
            WC = math.ceil(C * h)
            imgReSize = cv2.resize(imgCrop, (WC, imgSize))
            imgReSizeHeight, imgReSizeWidth = imgReSize.shape[:2]
            wGap = math.ceil((imgSize - WC) / 2)
            imgWhite[0:imgReSizeHeight, wGap:WC + wGap] = imgReSize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    if time.time() - last_save > save_interval:
        counter += 1
        cv2.imwrite(f"{folder}/Image_{counter}.jpg", imgWhite)
        last_save = time.time()
        print(f"Saved: Image_{counter}.jpg")

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
