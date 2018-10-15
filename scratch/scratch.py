import cv2
import numpy as np
import time
import sys
from collections import Counter
#import schedule

video = 'http://algovdms.dot.state.al.us:1935/mnt-live/mgm-cam-002b.stream/chunklist_w1577430371.m3u8'

cap = cv2.VideoCapture(video)
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = 0)

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))		#get the frame height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))		#get the frame width
default_bg = cv2.imread('default_bg_n.png')

ret,frame = cap.read()
start = int(time.time())

first_divider = frame.shape[0] // 2
second_divider = frame.shape[0] // 1.5
third_divider = frame.shape[0] // 3

lst = []

# ============================================================================

def drawLines(frame):
	cv2.line(frame, (0, first_divider),(frame.shape[1], first_divider), (0,0,255), 1)
	cv2.line(frame, (0, int(second_divider)),(frame.shape[1], int(second_divider)), (0,255,0), 1)
	cv2.line(frame, (0, int(third_divider)),(frame.shape[1], int(third_divider)), (255,0,0), 1)
	
	return frame

# ============================================================================	

def get_centroid(x, y, w, h):
	x1 = int(w / 2)
	y1 = int(h / 2)

	cx = x + x1
	cy = y + y1

	return (cx, cy)	

# ============================================================================

def most_common(lst):
	data = Counter(lst)
	
	return max(lst, key=data.get)
	
def filter(fgmask):
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	erosion = cv2.erode(fgmask, kernel, iterations = 2)
	opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, None)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, None)
	dilation = cv2.dilate(closing, kernel, iterations = 2)
	
	return dilation
	
# ============================================================================
	
def detect_vehicles(fgmask):
	cars = []
	mask = filter(fgmask)
	cv2.imshow('filter', mask)
	_, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for (i, contour) in enumerate(contours):
		(x, y, w, h) = cv2.boundingRect(contour)
		centroid = get_centroid(x, y, w, h)
		cars.append(((x, y, w, h), centroid))	
	
	return cars
	
# ============================================================================
	
def draw_vehicles(fgmask, frame):
	cars = detect_vehicles(fgmask)
	for (i, car) in enumerate(cars):
		contour, centroid = car
		x, y, w, h = contour
		cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), (255, 0, 0), 1)
		cv2.circle(frame, centroid, 2, (0, 0, 255), -1)

	return frame
	
# ============================================================================
	
def ROI(frame):
	#image = cv2.imread(get_latest_image('results'))
	image = cv2.imread('roi.png')
	imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 127, 255, 0)
	_, contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	stencil = np.zeros(frame.shape).astype(frame.dtype)
	cv2.fillPoly(stencil, contours, (255,255,255))
	roi = cv2.bitwise_and(frame, stencil)
	
	return roi

# ============================================================================	
	
'''def count(cars, frame):
	font = cv2.FONT_HERSHEY_SIMPLEX	
	count1 = 0
	count2 = 0
	count3 = 0
	for car in cars:
		if car[0][1] < first_divider:
			count1 += 1
		if car[0][1] < second_divider:
			count2 += 1
		if car[0][1] < third_divider:
			count3 += 1

	cv2.putText(frame, 'Line 1: ' + str(count1), (10, 35), font, 0.8, (0, 0, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)
	cv2.putText(frame, 'Line 2: ' + str(count2), (10, frame.shape[1] //2), font, 0.8, (255, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)
	cv2.putText(frame, 'Line 3: ' + str(count3), (10, frame.shape[1] //6), font, 0.8, (0, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)
	cv2.putText(frame, 'Detected Vehicles: ' + str(count3), (10, frame.shape[1] //6), font, 0.8, (0, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)
	
	return frame'''
	
# ============================================================================

'''def get_latest_image(img):
	list_of_files = glob.glob(img + '/*')
	latest_file = max(list_of_files, key = os.path.getctime)
	
	return latest_file'''
	
# ============================================================================
	
def main(start):
	while(1):
		ret,frame = cap.read()
		copy = frame.copy()
		
		#bg = ROI(default_bg)
		#fgbg.apply(bg, None, 1.0)
		
		#copy = drawLines(copy)
		roi = ROI(copy)
		fgmask = fgbg.apply(roi, None, 0.01)

		cars = detect_vehicles(fgmask)
		lst.append(len(cars))
		count = most_common(lst)
		mx = max(lst)
		font = cv2.FONT_HERSHEY_SIMPLEX
		'''if int(time.time()) - start > 1:
			count = most_common(lst)
			start = int(time.time())'''
		cv2.putText(copy, 'Count mc: ' + str(count), (10, copy.shape[1] //6), font, 0.8, (0, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)
		cv2.putText(copy, 'Count mx: ' + str(mx), (10, copy.shape[1] //3), font, 0.8, (0, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)
		cv2.putText(copy, 'Count: ' + str(len(cars)), (10, copy.shape[1] //2), font, 0.8, (0, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)
		
		output = draw_vehicles(fgmask, copy)
		
		#cv2.imshow('frame', copy)
		#cv2.imshow('mask', mask)
		cv2.imshow('roi', roi)
		#cv2.imshow('fgmask', fgmask)
		cv2.imshow('output', output)
		
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			#cv2.imwrite('default_bg1.png', frame)
			break

if __name__ == "__main__":
	main(start)

cap.release()
cv2.destroyAllWindows()