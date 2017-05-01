from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import RPi.GPIO as GPIO



def robotMover(x,y):


	pass





def tracker():
	camera = PiCamera()
	camera.rotation = 180
	camera.resolution = (640, 480)
	camera.framerate = 32
	rawCapture = PiRGBArray(camera, size=(640,480))

	time.sleep(0.1)
	frame = np.empty((480 * 640 * 3), dtype=np.uint8)
	camera.capture(frame, format="bgr")
	frame = frame.reshape((480,640,3))


	#setup initial location of window
	c,r,w,h = 270 ,150, 100,100
	track_window = (c,r,w,h)

	#Define a region of Interest
	roi = frame[r:r+h, c:c+w]

	#Change the color from bgr to hsv
	hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	#Define color boundaries
	#Test on tracking red objects
	lowerColor = np.array((0,55,100))
	upperColor = np.array((8,69,100))


	#Performs Actual Color Detection using the specified ranges
	mask = cv2.inRange(hsv_roi, lowerColor, upperColor)
	
	#Remove Inaccuracies from the mask
	#mask = cv2.erode(mask, None, iterations=2)	
	#mask = cv2.dilate(mask, None, iterations=2)	



	#format: cv2.calcHist(images,channels,mask,histSize, ranges)
	roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
	cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

	#Stop the algorithm when accuracy of 1 is reached or 50 iterations have been run
	term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1)


	for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
		frame =frame.array
	
		frame = frame.reshape((480,640,3))
	

		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	
		dst = cv2.calcBackProject([hsv], [0],roi_hist,[0,180], 1)
		
		#apply meanshift to get the new location
		ret, track_window = cv2.meanShift(dst, track_window, term_crit)

		#print ret
		x,y,w,h = track_window
	
		#Draw a rectangle on the image
		cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
	
		#Write 'Tracked' over the rectangle
		cv2.putText(frame, 'Tracked', (x + 5, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
		(255,255,255), 2, cv2.CV_AA)
		
		#Show image
		cv2.imshow("Tracking", frame)
	
		key = cv2.waitKey(60) & 0xFF

		rawCapture.truncate(0)
	
		#if key == ord("q"):
		#	break	
		#else:
		#	cv2.imwrite(chr(key) + ".jpg", frame)

	#camera.release()
	cv2.destroyAllWindows()
	#return [x, y]



tracker()




