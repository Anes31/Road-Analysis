import numpy as np
import cv2
import sys
import time
from PIL import Image

fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold = 200, detectShadows = 0)
#video = 'C:/Users/HP/Desktop/Computer Vision Project/Mask/Long video high resolution/video.mp4'
video = 'http://algovdms.dot.state.al.us:1935/mnt-live/mgm-cam-002b.stream/chunklist_w1577430371.m3u8'
cap = cv2.VideoCapture(video)

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))		#get the frame height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))		#get the frame width
#sys.stdout = open('output.txt','wt')		#create a file
x = 0
count = 0
#start = time.process_time()		#start time
start = time.time()		#start time

while(1 and time.time() - start < 300):		#stop the loop after 5mn
#while(1):
	ret,frame = cap.read()
	fgmask = fgbg.apply(frame)		#apply the MOG2 mask on the video
	
	if fgmask is not None:
		erode = cv2.erode(fgmask, None, iterations = 1)		#erosion to erase unwanted small contours
		opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, None)		#eliminate false positive on erosion

		#cv2.imshow('MOG2', fgmask)
		#cv2.imshow('erosion', erode)
		#cv2.imshow('opening', opening)		#show the frame after treating it
		#cv2.imshow('frame', frame)		#show the original frame
		t = int(time.time() - start)
		if t > 12:
			if x < 5:		#this condition is used because if the creation of the black image starts at the same time as the video all the pixels will be white
				blank_image = np.zeros((frame_height,frame_width,3), np.uint8)		#create a black image with the same height and width of the video
				blank_image1 = np.zeros((frame_height,frame_width,3), np.uint8)
				blank_image2 = np.zeros((frame_height,frame_width,3), np.uint8)
				x += 1
			else:
				blank_image[(opening==255) | (opening==127)] = 255
				blank_image1[(erode==255) | (erode==127)] = 255
				blank_image2[(fgmask==255) | (fgmask==127)] = 255
			if (count == 200):
				cv2.imwrite('final/0-original.png', frame)
			#print(time.time() - start)
			#cv2.imshow('blank_image', blank_image)
			#cv2.imshow('original', frame)
			#cv2.imshow('erode', blank_image1)
			#cv2.imshow('opening', blank_image)
			#cv2.imshow('fgmask', blank_image2)
			cv2.imwrite('original/frame%d.png' %t, frame)
			cv2.imwrite('unprocessed/frame%d.png' %t, blank_image2)
			cv2.imwrite('erodes/frame%d.png' %t, blank_image1)
			cv2.imwrite('frames/frame%d.png' %t, blank_image)		#saving the background treated
			count+=1
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()