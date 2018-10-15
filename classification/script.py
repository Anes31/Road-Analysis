# The source of this code is from stackoverflow and here's the link: https://stackoverflow.com/questions/36254452/counting-cars-opencv-python-issue/36274515#36274515
import logging
import logging.handlers
import os
import time
import sys

import cv2
import numpy as np

from vehicle_counter import VehicleCounter
import glob

# ============================================================================
cascade_src = 'cars.xml'
car_cascade = cv2.CascadeClassifier(cascade_src)
IMAGE_DIR = "images"
IMAGE_FILENAME_FORMAT = IMAGE_DIR + "/frame_%04d.png"

# Support either video file or individual frames
CAPTURE_FROM_VIDEO = False
if CAPTURE_FROM_VIDEO:
    IMAGE_SOURCE = 'http://algovdms.dot.state.al.us:1935/mnt-live/mgm-cam-002b.stream/chunklist_w1577430371.m3u8' # Video file
else:
    IMAGE_SOURCE = IMAGE_FILENAME_FORMAT # Image sequence

# Time to wait between frames, 0=forever
WAIT_TIME = 1 # 250 # ms

LOG_TO_FILE = True

# Colours for drawing on processed frames    
DIVIDER_COLOUR = (255, 255, 0)
BOUNDING_BOX_COLOUR = (255, 0, 0)
CENTROID_COLOUR = (0, 0, 255)

# ============================================================================

def init_logging():
    main_logger = logging.getLogger()

    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s] %(message)s'
        , datefmt='%Y-%m-%d %H:%M:%S')

    handler_stream = logging.StreamHandler(sys.stdout)
    handler_stream.setFormatter(formatter)
    main_logger.addHandler(handler_stream)

    if LOG_TO_FILE:
        handler_file = logging.handlers.RotatingFileHandler("debug.log"
            , maxBytes = 2**24
            , backupCount = 10)
        handler_file.setFormatter(formatter)
        main_logger.addHandler(handler_file)

    main_logger.setLevel(logging.DEBUG)

    return main_logger

# ============================================================================

def save_frame(file_name_format, frame_number, frame, label_format):
    file_name = file_name_format % frame_number
    label = label_format % frame_number

    log.debug("Saving %s as '%s'", label, file_name)
    cv2.imwrite(file_name, frame)

# ============================================================================

def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1

    return (cx, cy)

# ============================================================================

def detect_vehicles(fg_mask, processed):
    log = logging.getLogger("detect_vehicles")

    MIN_CONTOUR_WIDTH = 0
    MIN_CONTOUR_HEIGHT = 0

    # Find the contours of any vehicles in the image
    #_, contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)      
    
    #log.debug("Found %d vehicle contours.", len(contours))

    matches = []
    #cv2.imshow('roi', fg_mask)	
    gray = cv2.cvtColor(fg_mask, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x,y,w,h) in cars:
    #for (i, contour) in enumerate(contours):
        #(x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(processed,(x,y),(x+w,y+h),(0,0,255),2)    
        contour_valid = (w >= MIN_CONTOUR_WIDTH) and (h >= MIN_CONTOUR_HEIGHT)

        #log.debug("Contour #%d: pos=(x=%d, y=%d) size=(w=%d, h=%d) valid=%s"
        #    , i, x, y, w, h, contour_valid)

        if not contour_valid:
            continue

        centroid = get_centroid(x, y, w, h)

        matches.append(((x, y, w, h), centroid))

    return matches

# ============================================================================

'''def filter_mask(fg_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Fill any small holes
    closing = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    # Remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    # Dilate to merge adjacent blobs
    dilation = cv2.dilate(opening, kernel, iterations = 2)

    return closing'''

# ============================================================================

def process_frame(frame_number, frame, bg_subtractor, car_counter):
    log = logging.getLogger("process_frame")

    # Create a copy of source frame to draw into
    processed = frame.copy()
    #print(frame.shape[1], car_counter.divider)
    # Draw dividing line -- we count cars as they cross this line.
    cv2.line(processed, (0, int(car_counter.divider)), (frame.shape[1], int(car_counter.divider)), DIVIDER_COLOUR, 1)

    # Remove the background
    #fg_mask = bg_subtractor.apply(frame, None, 0.01)
    #fg_mask = filter_mask(fg_mask)

    #save_frame(IMAGE_DIR + "/mask_%04d.png"
    #    , frame_number, fg_mask, "foreground mask for frame #%d")

    image = cv2.imread(get_latest_image('results'))
    cont = contoursFrom(image)
    stencil = np.zeros(processed.shape).astype(processed.dtype)
    cv2.fillPoly(stencil, cont, (255,255,255))
    img = cv2.bitwise_and(processed, stencil)
	
    matches = detect_vehicles(img, processed)

    log.debug("Found %d valid vehicle contours.", len(matches))
    for (i, match) in enumerate(matches):
        contour, centroid = match

        log.debug("Valid vehicle contour #%d: centroid=%s, bounding_box=%s", i, centroid, contour)

        x, y, w, h = contour

        # Mark the bounding box and the centroid on the processed frame
        # NB: Fixed the off-by one in the bottom right corner
        #cv2.rectangle(processed, (x, y), (x + w - 1, y + h - 1), BOUNDING_BOX_COLOUR, 1)
        cv2.circle(processed, centroid, 2, CENTROID_COLOUR, -1)

    log.debug("Updating vehicle count...")
    count = car_counter.update_count(matches, processed)

    return processed

# ============================================================================
	
def get_latest_image(img):
    list_of_files = glob.glob(img + '/*')
    latest_file = max(list_of_files, key = os.path.getctime)
    
    return latest_file
	
# ============================================================================

def contoursFrom(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    _, contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours
	
# ============================================================================

# ============================================================================
	
def main():
    log = logging.getLogger("main")

    log.debug("Creating background subtractor...")
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows = 0)

    log.debug("Pre-training the background subtractor...")
    default_bg = cv2.imread('default_bg.png')
    bg_subtractor.apply(default_bg, None, 1.0)

    car_counter = None # Will be created after first frame is captured

    # Set up image source
    cap = cv2.VideoCapture('http://algovdms.dot.state.al.us:1935/mnt-live/mgm-cam-002b.stream/chunklist_w1577430371.m3u8')
    log.debug("Initializing video capture device #%s...", IMAGE_SOURCE)

    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    log.debug("Video capture frame size=(w=%d, h=%d)", frame_width, frame_height)

    log.debug("Starting capture loop...")
    frame_number = -1
    while True:
        frame_number += 1
        log.debug("Capturing frame #%d...", frame_number)
        ret, frame = cap.read()
        if not ret:
            log.error("Frame capture failed, stopping...")
            break

        log.debug("Got frame #%d: shape=%s", frame_number, frame.shape)

        if car_counter is None:
            # We do this here, so that we can initialize with actual frame size
            log.debug("Creating vehicle counter...")
            car_counter = VehicleCounter(frame.shape[:2], frame.shape[0] / 2)

        # Archive raw frames from video to disk for later inspection/testing
        if CAPTURE_FROM_VIDEO:
            save_frame(IMAGE_FILENAME_FORMAT
                , frame_number, frame, "source frame #%d")
		
        log.debug("Processing frame #%d...", frame_number)
        processed = process_frame(frame_number, frame, bg_subtractor, car_counter)
					
        save_frame(IMAGE_DIR + "/processed_%04d.png"
            , frame_number, processed, "processed frame #%d")

        #cv2.imshow('Source Image', frame)
        cv2.imshow('Processed Image', processed)

        log.debug("Frame #%d processed.", frame_number)       

        c = cv2.waitKey(WAIT_TIME)
        if c == 27:
            log.debug("ESC detected, stopping...")
            #cv2.imwrite('default_bg1.png', frame)
            break

    log.debug("Closing video capture device...")
    cap.release()
    cv2.destroyAllWindows()
    log.debug("Done.")

# ============================================================================

if __name__ == "__main__":
    log = init_logging()

    if not os.path.exists(IMAGE_DIR):
        log.debug("Creating image directory `%s`...", IMAGE_DIR)
        os.makedirs(IMAGE_DIR)

    main()