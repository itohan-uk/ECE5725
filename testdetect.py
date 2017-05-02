from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import imutils


			
		
		




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
	#c,r,w,h = 270 ,150, 100,100
	#track_window = (c,r,w,h)

	#Define a region of Interest
	#roi = frame[r:r+h, c:c+w]

	#Change the color from bgr to hsv
	#hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	#Define color boundaries
	#Test on tracking red objects
	lowerColor = np.array((0,55,100))
	upperColor = np.array((8,69,100))


	#Performs Actual Color Detection using the specified ranges
	#mask = cv2.inRange(hsv_roi, lowerColor, upperColor)
	
	#Remove Inaccuracies from the mask
	#mask = cv2.erode(mask, None, iterations=2)	
	#mask = cv2.dilate(mask, None, iterations=2)	



	#format: cv2.calcHist(images,channels,mask,histSize, ranges)
	#roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
	#cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

	#Stop the algorithm when accuracy of 1 is reached or 50 iterations have been run
	#term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1)


	for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
		frame =frame.array
	
		frame = frame.reshape((480,640,3))
		
		#Resize the frame
		#frame = imutils.resize(frame, width=600)
	
		#frame = cv2.GaussianBlur(frame, (21, 21), 0)

		#Convert to hsv
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		
		
		#Construct a mask for red and perform a series of iterations
		#and dilations
		mask = cv2.inRange(hsv, lowerColor, upperColor)
		#mask = cv2.erode(mask, None, iterations=2)	
		#mask = cv2.dilate(mask, None, iterations=2)	
		
		
		
		#find contours in the mask
		contours= cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, 
		cv2.CHAIN_APPROX_SIMPLE)[-2]
		
		
		
		
		
		if len(contours) > 0:
			
		
			#print contours
			#find the largest contour and use it to draw bounding rect
			maxContour = max(contours, key=cv2.contourArea)
			#print maxContour
			
			#compute the bounding box for the contour
			(x, y, w, h) = cv2.boundingRect(maxContour)
			print [x, y, w,h]
			#cv2.rectangle(frame, (x, y), (x + w , y + h), 255, 2)
			cv2.rectangle(frame, (x, y), (x + w , y - 100), 255, 2)
	
	
			#Write 'Tracked' over the rectangle
			cv2.putText(frame, 'Tracked', (x + 5, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
			(255,255,255), 2, cv2.CV_AA)
			
			#print "drawn"
			
			
		#else:
		#	print 'none'
			
			
			
			
				
		#Show image
		cv2.imshow("Tracking", frame)
	
		
		
		
		key = cv2.waitKey(60) & 0xFF
		
		
		#Clear current frame for next frame
		rawCapture.truncate(0)
	
		#if key == ord("q"):
		#	break	
		#else:
		#	cv2.imwrite(chr(key) + ".jpg", frame)

	#camera.release()
	cv2.destroyAllWindows()
	#return [x, y]



tracker()




