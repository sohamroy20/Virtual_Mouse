import cv2
import mediapipe as mp
import time
import HTmodule as htm
import numpy as np
import autopy
pTime=0
frameR=100
detector = htm.handDetect(maxHands=1)
wScr,hScr= autopy.screen.size()
wc,hc=640,480
cap = cv2.VideoCapture(0)
cap.set(3,wc)
cap.set(4,hc)
print(wScr,hScr)
x1,x2,y1,y2=0,0,0,0
fingers=[0,0,0,0,0]
while True:
    #hand landmarks
    success, img=cap.read()
    img = detector.findHand(img)
    lmlist,bbox= detector.findPos(img)
    ##get the tip of index and middle fingers

    if len(lmlist)!=0:
        x1,y1=lmlist[8][1:]
        x2,y2=lmlist[12][1:]

        #print(x1,y1,x2,y2)  #check which fingers are up

        fingers=detector.fingersUp()
        cv2.rectangle(img, (frameR, frameR), (wc - frameR, hc - frameR), (255, 0, 0), 2)  # frame reduction
    ##Moving mode- index finger only

        if fingers[1] == 1 and fingers[2] == 0:
            # convert coordinates to set

            x3 = np.interp(x1, (frameR, wc-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hc-frameR), (0, hScr))
            # Smothen Values
            # movemouse
            autopy.mouse.move(wScr - x3, y3)
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
        ##check clicking mode

        if fingers[1] == 1 and fingers[2] == 1:
            length,img, lineInfo = detector.findDistance(8,12,img)
            #print(length)
            if length<29:
                cv2.circle(img, (lineInfo[4],lineInfo[5]), 14, (0,255,0), cv2.FILLED) # Green for clicked
                autopy.mouse.click()    # mouse click

    #Find distance b/w fingers \/\/ DONE

    #check frame rate and dispaly
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(20,50),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,2,(0,0,255),3)
    cv2.imshow("img",img)
    cv2.waitKey(1)