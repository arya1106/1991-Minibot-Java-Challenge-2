import numpy as np
import cv2 as cv
import time
from cscore import CameraServer as cs
from networktables import NetworkTables, NetworkTablesInstance

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
x_res = cap.get(cv.CAP_PROP_FRAME_WIDTH)
y_res = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
ntinst = NetworkTablesInstance.getDefault()
ntinst.startClientTeam(0)
ntinst.startDSClient()
nt = NetworkTables.getTable('vision')

output_blue = cs.getInstance().putVideo('Blue', 320, 240)
output_green = cs.getInstance().putVideo('Green', 320, 240)
output_clear = cs.getInstance().putVideo('Clear', 320, 240)

time.sleep(0.5)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Our operations on the frame come here
    in_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV )
    outBlue = cv.inRange(in_frame, (100, 25, 40), (130, 255, 255))
    outGreen = cv.inRange(in_frame, (70, 40, 20), (100, 255, 255))
    _Blue, blueContours, blueHeirarchy = cv.findContours(outBlue, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    _Green, greenContours, greenHeirarchy = cv.findContours(outGreen, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    kernel = np.ones((3,3), np.uint16)
    outBlue = cv.morphologyEx(outBlue, cv.MORPH_OPEN, kernel=kernel, iterations=1)
    outGreen = cv.morphologyEx(outGreen, cv.MORPH_OPEN, kernel=kernel, iterations=1)
    center_x_steer = 0
    greenArea = 0
    stop = False
    if len(blueContours) > 0:
        largestBlue = blueContours[0]
        for contour in blueContours:
            if cv.contourArea(contour) > cv.contourArea(largestBlue):
                largestBlue = contour
        x, y, w, h = cv.boundingRect(largestBlue)
        outBlue = cv.rectangle(outBlue, (x,y), (x+w, y+h), (255,255,255), 1)
        outBlue = cv.circle(outBlue, (int(x+w/2),int(y+h/2)), radius=1, color=(0,0,255), thickness=-1)
        center_x_steer = ((x+w/2)-(x_res/2))/(x_res/2)
    if len(greenContours) > 0:
        largestGreeen = greenContours[0]
        for contour in greenContours:
            if cv.contourArea(contour) > cv.contourArea(largestGreeen):
                largestGreeen = contour
        x, y, w, h = cv.boundingRect(largestGreeen)
        outGreen = cv.rectangle(outGreen, (x,y), (x+w, y+h), (255,255,255), 1)
        greenArea = cv.contourArea(largestGreeen)
        if(greenArea>2000):
            stop = y>75
    nt.putNumber('center_x', center_x_steer)
    nt.putNumber('green_area', greenArea)
    nt.putNumber('y', y+h/y_res)
    nt.putBoolean('stop', stop)
    output_blue.putFrame(outBlue)
    output_green.putFrame(outGreen)
    output_clear.putFrame(frame)
    time.sleep(0.01)
