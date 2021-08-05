import cv2
import mediapipe as mp
import time
import HTmodule as htm
import numpy as np
import autopy
pTime=0

detector = htm.handDetect(maxHands=1)
wc,hc=640,480
cap = cv2.VideoCapture(0)
cap.set(3,wc)
cap.set(4,hc)

while True:
    #hand landmarks
    success, img=cap.read()
    img = detector.findHand(img)
    lmlist,bbox= detector.findPos(img)
    ##get the tip of index and middle fingers

    if len(lmlist)!=0:
        x1,y1=lmlist[8][1:]
        x2,y2=lmlist[12][1:]

        #print(x1,y1,x2,y2)

        ##check which fingers are up
        fingers=detector.fingersUp()
        print(fingers)
    ##Moving mode-
        #convert coordinates to set
    #Smothen Values
    #movemouse
    ##check clicking mode
    #Find distance b/w fingers
    #mouse clickss if dist is less
    #check frame rate and dispaly
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(20,50),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,2,(0,0,255),3)
    cv2.imshow("img",img)
    cv2.waitKey(1)