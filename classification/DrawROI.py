import numpy as np
import cv2
import os, os.path
import glob
		
def draw_poly(img, contours):
	cv2.polylines(img, contours, True, (0,0,255), 3)
	
	return img

def contours(img):
	imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 127, 255, 0)
	_, contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	return contours
	
def processed_img(image, threshold1, threshold2):
	resize = cv2.resize(image, (600, 400))
	processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	processed_img = cv2.GaussianBlur(processed_img, (1,1), 0)
	processed_img = cv2.Canny(processed_img, threshold1, threshold2)
	cv2.imshow('Canny', processed_img)

	return processed_img
	
def hough_lines(image):
	#lines = cv2.HoughLinesP(image, 5, np.pi/180, 1, np.array([]), 5, 10)
	line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
	
	return line_img
	
def get_latest_image(img):
	list_of_files = glob.glob(img + '/*')
	latest_file = max(list_of_files, key = os.path.getctime)
	return latest_file

def main():
	
	while(1):
		image = cv2.imread(get_latest_image('results'))
		#image = cv2.imread('TestFrame.png')
		imageOrig = cv2.imread('final/0-original.png')
		img = processed_img(image, 0, 0)
		cont = contours(image)
		line_img = hough_lines(img)
		processed = draw_poly(line_img, cont)
		processed1 = draw_poly(imageOrig, cont)
		cv2.imshow('Display', processed)
		cv2.imshow('ROI', processed1)
		#cv2.imwrite('final/ResultdrawROI.png', processed)
		erode = cv2.imread(get_latest_image('erodes'))
		opening = cv2.imread(get_latest_image('frames'))
		fgmask = cv2.imread(get_latest_image('unprocessed'))
		cv2.imwrite('final/2-erode.png', erode)
		cv2.imwrite('final/3-opening.png', opening)
		cv2.imwrite('final/1-unprocessed.png', fgmask)
		cv2.imwrite('final/5-ResultROI.png', processed1)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			cv2.destroyAllWindows()
			break

if __name__ == '__main__':
	main()