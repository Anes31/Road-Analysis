import numpy as np
import cv2

panel = np.zeros([100, 700], np.uint8)
cv2.namedWindow("panel")

def nothing(x):
	pass
	
cv2.createTrackbar("Threshold1", "panel", 0, 1000, nothing)
cv2.createTrackbar("Threshold2", "panel", 0, 1000, nothing)

def draw_lines(img, lines):
	try:
		for line in lines:
			for x1,y1,x2,y2 in line:
				cv2.line(img, (x1, y1), (x2, y2), [0,255,0], 6)
	except:
		pass

def roi(img, vertices):
	mask = np.zeros_like(img)
	cv2.fillPoly(mask, vertices, 255)
	masked = cv2.bitwise_and(img, mask)
	return masked

def contours(background):
	imgray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 127, 255, 0)
	_, contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	return contours
	
def canny(image, threshold1, threshold2):
	canny_img = cv2.GaussianBlur(image, (1,1), 0)
	canny_img = cv2.Canny(canny_img, threshold1, threshold2)
	cv2.imshow('Canny', canny_img)
	return canny_img
	
def processed_img(image, image1):
	processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	vertices = contours(image1)
	processed_img = roi(processed_img, vertices)
	cv2.imshow('Processed_img', processed_img)
	return processed_img
	
def hough_lines(image):
	lines = cv2.HoughLinesP(image, 5, np.pi/180, 15, np.array([]), 65, 10)
	line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
	draw_lines(line_img, lines)
	
	return line_img
	

def main():
	while(1):
		image = cv2.imread('emptyRoad.png')
		image1 = cv2.imread('contoursFilled.png')
		t1 = cv2.getTrackbarPos("Threshold1", "panel")
		t2 = cv2.getTrackbarPos("Threshold2", "panel")
		img1 = processed_img(image, image1)
		img = canny(img1, t1, t2)
		#img = img[:, :, 0]
		processed = hough_lines(img)
		cv2.imshow('Display', processed)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			cv2.destroyAllWindows()
			break

if __name__ == '__main__':
	main()