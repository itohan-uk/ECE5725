from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import imutils
from scipy.linalg import norm
			
		
def normalize(arr):
	rng = arr.max() - arr.min()
	amin = arr.min()
	return (arr-amin)*255/rng	




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
	#print threshOld 
	foundObject = 0
	#cannyV = 0
	for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
		frame =frame.array
	
		frame = frame.reshape((480,640,3))
	
		grayNew = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		grayNew = cv2.GaussianBlur(grayNew, (21, 21), 0)
		#grayNew = cv2.Canny(frame, 75, 200)
		frameDiff = cv2.absdiff(grayNew, grayOld)
		#thresh = cv2.threshold(frameDiff, 25, 255, cv2.THRESH_BINARY)[1]
		#diff = (normalize(thresh)) - (normalize(threshOld))
		
		#zeroNorm = norm(diff.ravel(), 0)
	
		if not foundObject:
			#print 'here'
			thresh = cv2.threshold(frameDiff, 135, 255, cv2.THRESH_BINARY)[1]
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
			
			#contours = np.abs(contours)
			#print contours
			#if len(contours) > 0:
				#print 'yup'
				#for c in contours:
				#	(x,y,w,h) = cv2.boundingRect(c)
				#	cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0,255), 2)
			cv2.drawContours(frame, contours, -1, (0,0,255), 2)
				#maxC = max(contours, key=cv2.contourArea)
				#(x,y,w,h) = cv2.boundingRect(maxC)
				#cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0,255), 2)
				
			
				
			#print 'yes'
				
		
		
		#print thresh
		#print 'here'
		#print threshOld
		
		print zeroNorm
		#print foundObject
		
		
		
			
		#print grayOld
		#print 'here'
		#print grayNew
		#break
		#frame = cv2.GaussianBlur(frame, (7, 7), 0)
		#frame = cv2.Canny(filtered, 75, 200)
		
		
		
		
		
		#Construct a mask for red and perform a series of iterations
		#and dilations
		#mask = cv2.inRange(hsv, lowerColor, upperColor)
		#mask = cv2.erode(mask, None, iterations=2)	
		#mask = cv2.dilate(mask, None, iterations=2)	
		
		
		
		#find contours in the mask
		#(contours, _)= cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, 
		#cv2.CHAIN_APPROX_SIMPLE)
		#contourList = []
		#for c in contours:
		#	approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True)
		#	area = cv2.contourArea(c)
		#	if ((len(approx) >= 8) & (area > 30)):
				#contourList.append(c)
				
		#cv2.drawContours(filtered, contourList, -1, (0,0,255), 2)
				
		
		
		
		
		#if len(contours) > 0 and not foundObject:
			
		
			#print contours
		
			#find the largest contour and use it to draw bounding rect
			#maxContour = max(contours, key=cv2.contourArea)
			#print maxContour
			
			#compute the bounding box for the contour
			#(x, y, w, h) = cv2.boundingRect(maxContour)
			#M = cv2.moments(maxContour)
			
			#try:
			#	center = (int(M["m10"]/M["m00"]), int(M["m01"]/ M["m00"]))
			#	print center 
			#except ZeroDivisionError:
			#	print 'error'
				
			#print [x, y, w,h]
			#cv2.rectangle(frame, (x, y), (x + w , y + h), 255, 2)
			#cv2.drawContours(frame, contours, -1, (0,255,0), 2)
			#if center:
			#	cv2.rectangle(frame, (center), (center[0] + w , center[1] - 100), 255, 2)
			#	cv2.putText(frame, 'Tracked', (center[0] + 5, center[1]-12), 
			#	cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.CV_AA)
			
			#else:
			#cv2.rectangle(frame, (x, y), (x + w , y - 100), 255, 2)
	
	
			#Write 'Tracked' over the rectangle
			#cv2.putText(frame, 'Tracked', (x + 5, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
			#(255,255,255), 2, cv2.CV_AA)
			#foundObject = 1
			
			
			#print "drawn"
			
			
		#else:
		#	print 'none'
			
			
		#if foundObject:
		#	cv2.rectangle(frame, (x, y), (x + w , y - 100), 255, 2)
	
	
			#Write 'Tracked' over the rectangle
		#	cv2.putText(frame, 'Tracked', (x + 5, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
		#	(255,255,255), 2, cv2.CV_AA)
				
			
				
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
	#return [x, y]



tracker()





