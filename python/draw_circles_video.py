#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__      = "Matheus Dib, Fabio de Miranda"


import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import sys
import auxiliar as aux


# If you want to open a video, just change v2.VideoCapture(0) from 0 to the filename, just like below
#cap = cv2.VideoCapture('hall_box_battery.mp4')

magenta=np.array([210,255,127])
hsv1_m, hsv2_m = aux.ranges(magenta)

ciano=np.array([127,255,127])
hsv1_c, hsv2_c = aux.ranges(ciano)

if len(sys.argv) > 1:
    arg = sys.argv[1]
    try:
        input_source=int(arg) # se for um device
    except:
        input_source=str(arg) # se for nome de arquivo
else:   
    input_source = 0

cap = cv2.VideoCapture(input_source)


# Parameters to use when opening the webcam.


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

lower = 0
upper = 1

print("Press q to QUIT")

# Returns an image containing the borders of the image
# sigma is how far from the median we are setting the thresholds
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged



while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # A gaussian blur to get rid of the noise in the image
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #blur = gray
    # Detect the edges present in the image
    bordas = auto_canny(blur)


    circles = []


    # Obtains a version of the edges image where we can draw in color
    #bordas_color = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
    #bordas_hsv = cv2.cvtColor(bordas_color, cv2.COLOR_BGR2HSV)

    # HoughCircles - detects circles using the Hough Method. For an explanation of
    # param1 and param2 please see an explanation here http://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
    circles = None
    circles = cv2.HoughCircles(bordas,cv2.HOUGH_GRADIENT,2,40,param1=50,param2=100,minRadius=5,maxRadius=60)

    if circles is not None:        
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            print(i)

            hsv1 = np.array([ 143, 50, 50], dtype=np.uint8)
            hsv2 = np.array([ 165, 255, 255], dtype=np.uint8)

            hsv3 = np.array([ 80, 50, 50], dtype=np.uint8)
            hsv4 = np.array([ 100, 255, 255], dtype=np.uint8)

            mask_magenta = cv2.inRange(hsv, hsv1, hsv2) 
            mask_cyan = cv2.inRange(hsv, hsv3, hsv4)

            mask_mag_blur = cv2.blur(mask_magenta, (3, 3))
            mask_cya_blur = cv2.blur(mask_cyan, (3, 3))

            kernal = np.ones((5, 5), "uint8") 

            mask_magenta = cv2.dilate(mask_mag_blur, kernal)

            mask_cyan = cv2.dilate(mask_cya_blur, kernal)

            segmentado_magenta = cv2.morphologyEx(mask_mag_blur,cv2.MORPH_CLOSE,np.ones((10, 10)))
            selecao_magenta = cv2.bitwise_and(frame, frame, mask=segmentado_magenta)

            segmentado_cyan = cv2.morphologyEx(mask_cya_blur,cv2.MORPH_CLOSE,np.ones((10, 10)))
            selecao_cyan = cv2.bitwise_and(frame, frame, mask=segmentado_cyan)

            # Creating contour to track magenta
            contours, hierarchy = cv2.findContours(mask_mag_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
      
            for pic, contour in enumerate(contours): 
                area = cv2.contourArea(contour) 
                if(area > 300): 
                    # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
                    cv2.circle(blur,(i[0],i[1]),i[2],(0,0,255),2)
                    # draw the center of the circle
                    cv2.circle(blur,(i[0],i[1]),2,(0,255,0),3)
                    
                    cv2.putText(blur, "Magenta", (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0)) 

                else:
                    pass
            
            # Creating contour to track cyan
            contourn, hierarchy = cv2.findContours(mask_cya_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
      
            for pic, contour in enumerate(contourn): 
                area = cv2.contourArea(contour) 
                if(area > 300): 
                    # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
                    cv2.circle(blur,(i[0],i[1]),i[2],(255,0,0),2)
                    # draw the center of the circle
                    cv2.circle(blur,(i[0],i[1]),2,(0,255,0),3)
                    
                    cv2.putText(blur, "Ciano", (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0)) 
                
                else:
                    pass
            
            # draw the outer circle
            # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
            #cv2.circle(bordas_color,(i[0],i[1]),i[2],(0,0,255),2)
            # draw the center of the circle
            #cv2.circle(bordas_color,(i[0],i[1]),2,(0,255,0),3)

            #fonte = cv2.FONT_HERSHEY_SIMPLEX
            #cv2.putText(bordas_color,'Circulo',(i[0],i[1]), fonte, 1,(255,255,255),2, cv2.LINE_AA)

            # draw the outer circle
            #cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
            #cv2.circle(bordas_color,(i[0],i[1]),i[2],(0,255,0),2)
            
            # draw the center of the circle
            #cv2.circle(bordas_color,(i[0],i[1]),2,(0,0,255),3)
    


            
            #font = cv2.FONT_HERSHEY_SIMPLEX
            #cv2.putText(bordas_color,'Circulo',(i[0],i[1]), font, 1,(255,255,255),2, cv2.LINE_AA)

    # Draw a diagonal blue line with thickness of 5 px
    # cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    # cv2.line(bordas_color,(0,0),(511,511),(255,0,0),5)

    # cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    #cv2.rectangle(bordas_color,(384,0),(510,128),(0,255,0),3)

    # cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blur,'Press q to quit',(0,50), font, 1,(255,255,255),2,cv2.LINE_AA)

    #More drawing functions @ http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html

    # Display the resulting frame
    cv2.imshow('Detector de circulos',blur)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
