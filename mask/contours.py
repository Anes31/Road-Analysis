import numpy as np
import cv2
import os, os.path
import heapq

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
	area = []
	threshold_area = 0
	max_index = len(contours) - 1
	while counter <= max_index:
		area.append(cv2.contourArea(contours[counter]))
		counter += 1
	if len(area) > 1:
		largest_area = heapq.nlargest(2, area)
		diff = largest_area[0] - largest_area[1]
		avg = (largest_area[0] + largest_area[1])/2
		prcntDiff = diff/avg*100
		if prcntDiff > 80:
			threshold_area = largest_area[0]
		else:
			threshold_area = largest_area[1]
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
			cv2.drawContours(img, contours, savedContour, (255,255,255), -1)
		counter = counter + 1
	return img

def comparison(prevImg ,currentImg):
	prevImg = cv2.imread(prevImg)
	currentImg = cv2.imread(currentImg)
	"""diff = currentImg - prevImg
	avg = (currentImg + prevImg)/2
	prcntDiff = diff/avg*100 """
	diff = np.mean(prevImg != currentImg)
	return diff
	
def main():
	list = os.listdir('frames')
	files = len(list)
	while(1):
		for f in range(1, files+1):			
			img = cv2.imread('frames/frame{}.png'.format(f))
			if img is not None:
				img0 = processImage(img, 1)
				cont = findCont(img0)
				thresh = findThreshold(cont)
				clean = drawCont(img0, cont, thresh)
				img1 = processImage(clean, 5)
				cont1 = findCont(img1)
				thresh1 = findThreshold(cont1)
				result = drawROI(img1, cont1, thresh1)
				#cv2.imwrite('alg2/img/img{}.png'.format(f), img)
				#cv2.imwrite('alg2/img0/img0{}.png'.format(f), img0)
				#cv2.imwrite('alg2/img1/img1{}.png'.format(f), img1)
				#cv2.imwrite('alg2/result/result{}.png'.format(f), result)
				cv2.imwrite('final/4-dilation.png', img1)
				cv2.imwrite('results/contoursFilled{}.png'.format(f), result)
				if f >= 20 and comparison('frames/frame{}.png'.format(f-10), 'frames/frame{}.png'.format(f))>0.2:
					print('Please check frame{} and frame{}'.format(f-10, f))
					continue
				#cv2.imshow('Display', result)
				continue
		break
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			cv2.destroyAllWindows()
			break
if __name__ == '__main__':
	main()