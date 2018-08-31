import cv2
import numpy as np

image = cv2.imread('/home/pi/Desktop/road.png')
points = [(0,0),(0,0),(0,0),(0,0)]
index = 0



def getpoints(event,x,y,flags,param):
    global image
    global index
    global points
    if event == cv2.EVENT_LBUTTONDBLCLK:
        points[index]= (x,y)
        index +=1
cv2.namedWindow('image')
cv2.setMouseCallback('image',getpoints)
booli = True
while(booli ==True):
    cv2.imshow('image',image)
    key = cv2.waitKey(20) & 0xFF
    if key == '27': 
        cv2.destroyAllWindows()
    if (index==4):
        booli = False

print(points)
file = open('points.txt','w')
file.write(str(points))
file.close()
print(points)
