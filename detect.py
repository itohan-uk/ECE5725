from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import imutils
from scipy.linalg import norm
			


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


	
	oldFrame = frame
	grayOld =  cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)
	grayOld = cv2.GaussianBlur(grayOld, (21, 21), 0)
	grayNew = grayOld
	frameDiff = cv2.absdiff(grayNew, grayOld)
	threshOld = cv2.threshold(frameDiff, 25, 255, cv2.THRESH_BINARY)[1]
	
	foundObject = 0
	#cannyV = 0
	for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
		frame =frame.array
	
		frame = frame.reshape((480,640,3))
	
		grayNew = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		grayNew = cv2.GaussianBlur(grayNew, (21, 21), 0)
		frameDiff = cv2.absdiff(grayNew, grayOld)
	
		if not foundObject:
		
			thresh = cv2.threshold(frameDiff, 100, 255, cv2.THRESH_BINARY)[1]
			zeroNorm = abs(norm(thresh) - norm(threshOld))
			#dilateThresh = cv2.dilate(thresh, None, iterations=2)
			
		#print '_________________________'
		#print thresh
		#print threshOld
		
		if zeroNorm:
			foundObject = 1
			
			canny =cv2.Canny(thresh, 50, 100)
			#cannyV = 1
			contours, _ = cv2.findContours(canny, cv2.RETR_TREE, 
			cv2.CHAIN_APPROX_SIMPLE)
			maxC = max(contours, key=cv2.contourArea)
			#print maxC
			#if cv2.contourArea(maxContour > 20):
		
			
			cv2.drawContours(frame, contours, -1, (0,0,255), 2)
			
			M = cv2.moments(maxC)
			cX, cY =(int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))	
			
				
			#TO DO: write a function that moves the robot around until its
			#center is close to CX, CY, then it should start working,
			#reset zeroNorm and found object when it picks the object and 
			#puts it in its location
				
				#(x,y,w,h) = cv2.boundingRect(maxC)
				#cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0,255), 2)
				
			
				
			#print 'yes'
				
		
		
		#print thresh
		#print 'here'
		#print threshOld
		
		print zeroNorm
		#print foundObject
		
		
		
		
				
			
				
		#Show image
		#if cannyV:
		#	cv2.imshow("Tracking", canny)
			
			
		#else:	
		cv2.imshow("Tracking", frame)
		oldFrame = frame
		grayOld =  cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)
		grayOld = cv2.GaussianBlur(grayOld, (21, 21), 0)
		threshOld = thresh
		#grayOld = cv2.Canny(oldFrame, 75, 200)
		
		
		
		key = cv2.waitKey(60) & 0xFF
		
		
		#Clear current frame for next frame
		rawCapture.truncate(0)
	
		if key == ord("q"):
			sys.exit()
		#else:
		#	cv2.imwrite(chr(key) + ".jpg", frame)

	camera.release()
	cv2.destroyAllWindows()




tracker()





