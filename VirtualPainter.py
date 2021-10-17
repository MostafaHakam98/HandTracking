import cv2
import numpy as np
import os
import handTrackingModule as htm

##################

brushThickness = 15

##################

folderPath = "Header"
myList = os.listdir(folderPath)
overlayList = []

#print(myList)

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
    
#print(len(overlayList))
header = overlayList[0]
color = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon = 0.9)

xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    # 1. Import Image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)
    
    if len(lmList) != 0:
        #print(lmList)
        
        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        
    
        # 3. Check which finders are up
        fingers = detector.fingersUp()
        #print(fingers)
    
        # 4. If Selection Mode - Two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25),
                           color, cv2.FILLED)
            print("Selection Mode")
            
            # Check for the click
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    color = (255, 0, 255)
                    brushThickness = 15
                
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    color = (255, 0, 0)
                    brushThickness = 15
                
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    color = (0, 255, 0)
                    brushThickness = 15
                
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    color = (0, 0, 0)
                    brushThickness = 60
        
        # 5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15,
                           color, cv2.FILLED)
            print("Drawing Mode")
            
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
                
            cv2.line(img, (xp, yp), (x1, y1), color, brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), color, brushThickness)
            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
                
    # Setting Header Image
    img[0:125, 0:1280] = header
    
    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    
    cv2.imshow("Web Cam", img)
    cv2.waitKey(1)