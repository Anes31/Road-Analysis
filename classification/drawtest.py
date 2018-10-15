import cv2
import numpy as np
	
while(1):
	image = cv2.imread('final/4-dilation.png')
	imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 127, 255, 0)
	_, contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	orgnl = cv2.imread('final/0-original.png')
	stencil = np.zeros(orgnl.shape).astype(orgnl.dtype)
	cv2.fillPoly(stencil, contours, (255,255,255))
	res = cv2.bitwise_and(orgnl, stencil)
	cv2.imshow('res', res)
	
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()