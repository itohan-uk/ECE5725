from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
#import imutils
from scipy.linalg import norm
import RPi.GPIO as GPIO



################# MOTOR STUFF ###################			
timeStart = time.clock()
timeElapsed = time.clock() - timeStart

GPIO.setmode(GPIO.BCM)
GPIO.setup(5, GPIO.OUT)
GPIO.setup(6, GPIO.OUT) 
pW = 0.0015
#pW2 = .0015

freq = 50

#choice=0
#choice2=0

dc = (pW * freq) * 100
#dc2 = (pW2 * freq) * 100

p = GPIO.PWM(5,freq)
p2 = GPIO.PWM(6, freq) 

p.start(dc)
p2.start(dc)


dcCount = 1
#clockwise time variables
c_time_L = time.clock() - timeStart
c_time_R = time.clock() - timeStart 

#counter clockwise time variables
cc_time_L = time.clock() - timeStart
cc_time_R = time.clock() - timeStart


def clockWise_L():
	global freq
	#global pW
	global dc
	global dcCount
	global choice
	global c_time_L

	pW = .0014
	dc = (pW * freq) * 100
	p.ChangeDutyCycle(dc)
	choice = choice+1
	c_time_L = time.clock() - timeStart
	
	

def clockWise_R():
	global freq
	#global pW2
	global dc2
	global dcCount
	global choice2
	global c_time_R

	pW2 = .0014
	dc2 = (pW2 * freq) * 100
	p2.ChangeDutyCycle(dc2)
	choice2 = choice2 + 1
	c_time_R = time.clock() - timeStart
	
def counterClockWise_L():
	global freq
	#global pW
	global dc
	global dcCount
	global choice
	global cc_time_L

	pW = .0016
	dc = (pW * freq) * 100
	p.ChangeDutyCycle(dc)
	choice = choice+1
	cc_time_L = time.clock() - timeStart

def counterClockWise_R():
	global freq
	#global pW2
	global dc2
	global dcCount
	global choice2
	global cc_time_R

	pW2 = .0016
	dc2 = (pW2 * freq) * 100
	p2.ChangeDutyCycle(dc2)
	choice2 = choice2 +1
	cc_time_R = time.clock() - timeStart


def immediateStop():
	p2.ChangeDutyCycle(0)
	p.ChangeDutyCycle(0)

def resume():
	global dc
	global dc2

	p.ChangeDutyCycle(dc)
	p2.ChangeDutyCycle(dc2)




def forward():
	global firstTime
	firstTime = time.clock()
	clockWise_R()
	counterClockWise_L()


def backward():
	clockWise_L()
	counterClockWise_R()


def pivotLeft():
	clockWise_R()
	clockWise_L()

def pivotRight():
	counterClockWise_R()
	counterClockWise_L()





################# END MOTOR STUFF ###################

			
			
def makeCenter(x):
	pass
	#centerX = 320
	
	
	
	
	
	
	
				
			


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
	#firstFrame = frame
	grayOld =  cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)
	firstGray = grayOld
	#grayOld = cv2.GaussianBlur(grayOld, (21, 21), 0)
	grayNew = grayOld
	frameDiff = cv2.absdiff(grayNew, grayOld)
	threshOld = cv2.threshold(frameDiff, 25, 255, cv2.THRESH_BINARY)[1]
	
	foundObject = 0
	#cannyV = 0
	newNorm = 1
	#contours = [1,1]
	drawNow = 0
	
	for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
		frame =frame.array
	
		frame = frame.reshape((480,640,3))
	
		grayNew = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#grayNew = cv2.GaussianBlur(grayNew, (21, 21), 0)
		frameDiff = cv2.absdiff(grayNew, grayOld)
		thresh = cv2.threshold(frameDiff, 50, 255, cv2.THRESH_BINARY)[1]
	
		
		if not foundObject:
			#print 'first'
		
			#thresh = cv2.threshold(frameDiff, 100, 255, cv2.THRESH_BINARY)[1]
			zeroNorm = abs(norm(thresh) - norm(threshOld))
			
			#dilateThresh = cv2.dilate(thresh, None, iterations=2)
			
		#print '_________________________'
		#print thresh
		#print threshOld
		
		if zeroNorm and newNorm:
			
			foundObject = 1
			#thresh = cv2.threshold(frameDiff, 100, 255, cv2.THRESH_BINARY)[1]
			newNorm = abs(norm(thresh) - norm(threshOld))
			#print 'newnorm ' + str(newNorm)
			#oldContours = contours
			
			#print '///oldContours///'
			#print oldContours
			#print '////////////////'
			#canny =cv2.Canny(thresh, 50, 100)
			#cannyV = 1
			#contours, _ = cv2.findContours(canny, cv2.RETR_TREE, 
			#cv2.CHAIN_APPROX_SIMPLE)
			
			#print '///Contours///'
			#print contours
			#print '////////////////'
			
			
			#try:
			#if len(contours):
				#oldContours = contours
			#	maxC = max(contours, key=cv2.contourArea)
			#	M = cv2.moments(maxC)
				
			#	try:
			#		cX, cY =(int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))	
			#	except ZeroDivisionError:
			#		continue
			
			
		#Draw when image stops changing
		if not newNorm and not drawNow:
			assert newNorm == 0.0
			
			
			#When it stops changing compare with first frame
			changeDiff = cv2.absdiff(grayNew, firstGray)
			threshChange = cv2.threshold(changeDiff, 50, 255, cv2.THRESH_BINARY)[1]
			cannyChange =cv2.Canny(threshChange, 50, 100)
		
			changeContours, _ = cv2.findContours(cannyChange, cv2.RETR_TREE, 
			cv2.CHAIN_APPROX_SIMPLE)
			
			
			#cv2.drawContours(frame, changeContours, -1, (0,0,255), 2)
			
			
			#Calculate Centroid
			maxC = max(changeContours, key=cv2.contourArea)
			M = cv2.moments(maxC)
			cX, cY =(int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
			
			drawNow = 1	
			
		if drawNow:
			cv2.drawContours(frame, changeContours, -1, (0,0,255), 2)
			
			#Move Object to Make cX equal to its Center
			
			
			
			
			
			#print cX
			#print cY
					
				
				
				
				
				
				
				
			#print maxC
			#if cv2.contourArea(maxContour > 20):
		
			#elif drawNow == 0 and len(contours)==0:# or drawNow == 1:
			#	drawNow = 1
				
				
			#if drawNow == 1:	
				#print 'drawing'
			#	cv2.drawContours(frame, oldContours, -1, (0,0,255), 2)
				#drawNow = 2
			
			#cv2.drawContours(frame, contours, -1, (0,0,255), 2)
			
			#M = cv2.moments(maxC)
			#cX, cY =(int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))	
			
			#except ValueError:
			#	continue
				
			#except ZeroDivisionError:
			#	continue	
				
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
		
		#print zeroNorm
		#print foundObject
		print 'zeroNorm ' + str(zeroNorm) + '\n'
		#print 'newNorm ' + str(newNorm) + '\n'
		
		
		
		
				
			
				
		#Show image
		#if cannyV:
		#	cv2.imshow("canny", canny)
			
			
		#else:	
		cv2.imshow("Tracking", frame)
		#cv2.imshow("thresh", thresh)
		oldFrame = frame
		grayOld =  cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)
		#grayOld = cv2.GaussianBlur(grayOld, (21, 21), 0)
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





