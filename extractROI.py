import numpy as np
import cv2
import sys
import time
from PIL import Image


def processVideo(cap):
	fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = 0)
	frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))		#get the frame height
	frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))		#get the frame width
	x = 0
	start = time.time()		#start time

	while(1 and time.time()-start < 300):		#stop the loop after 5mn
		ret,frame = cap.read()
		fgmask = fgbg.apply(frame)		#apply the MOG2 mask on the video

		erode = cv2.erode(fgmask, None, iterations = 1)		#erosion to erase unwanted small contours
		opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, None)		#eliminate false positive on erosion

		cv2.imshow('opening', opening)		#show the frame after treating it
		cv2.imshow('frame', frame)		#show the original frame

		if x < 5:		#this condition is used because if the creation of the black image starts at the same time as the video all the pixels will be white
			blank_image = np.zeros((frame_height,frame_width,3), np.uint8)		#create a black image with the same height and width of the video
			x += 1
		else:
			blank_image[(opening==255) | (opening==127)] = 255
		cv2.imshow('blank_image', blank_image)
		
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			cv2.imwrite('emptyRoad.png', frame)
			break
	cap.release()
	cv2.destroyAllWindows()
	return blank_image
	
def processImage(img, iteration):
	kernel = np.ones((5,5),np.uint8)
	dilation = cv2.dilate(img,kernel,iterations = iteration)
	return dilation
	
def findCont(img):
	imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 127, 255, 0)
	im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	return contours
	
def findThreshold(contours):
	counter = 0
	threshold_area = 0
	max_index = len(contours) - 1
	while counter <= max_index:
		area = cv2.contourArea(contours[counter])
		if area > threshold_area: 
			threshold_area = area
		counter = counter + 1
	return threshold_area
	
def drawCont(img, contours, threshold_area):
	counter = 0
	max_index = len(contours) - 1
	while counter <= max_index:
		area = cv2.contourArea(contours[counter])
		if area < threshold_area:
			savedContour = counter
			cv2.drawContours(img, contours, savedContour, (0,0,0), -1)
		counter = counter + 1
	return  img
		
def drawROI(img, contours, threshold_area):
	counter = 0
	max_index = len(contours) - 1
	while counter <= max_index:
		area = cv2.contourArea(contours[counter])
		if area < threshold_area:
			savedContour = counter
			#print (savedContour)
			cv2.drawContours(img, contours, savedContour, (255,255,255), -1)
		counter = counter + 1
	return img

def main():
	video = 'http://algovdms.dot.state.al.us:1935/mnt-live/mgm-cam-002b.stream/chunklist_w1577430371.m3u8'
	cap = cv2.VideoCapture(video)
	img = processVideo(cap)
	while(1):
		img0 = processImage(img, 1)
		cont = findCont(img0)
		thresh = findThreshold(cont)
		clean = drawCont(img0, cont, thresh)
		img1 = processImage(clean, 2)
		cont1 = findCont(img1)
		thresh1 = findThreshold(cont1)
		result = drawROI(img1, cont1, thresh1)
		cv2.imshow('processedImage', img0)
		cv2.imshow('clean', img1)
		cv2.imshow('Display', result)
		#cv2.imwrite('contoursFilled.png', result)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()