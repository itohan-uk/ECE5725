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
pW2 = .0015

freq = 50

dc = (pW * freq) * 100
dc2 = (pW2 * freq) * 100

p = GPIO.PWM(5,freq)
p2 = GPIO.PWM(6, freq) 

#p.start(dc)
#p2.start(dc)

p.start(0)
p2.start(0)




def clockWise_L():
	global freq

	pW = 0.0014
	dc = (pW * freq) * 100
	p.ChangeDutyCycle(dc)

	

def clockWise_R():
	global freq


	pW2 = 0.0014
	dc2 = (pW2 * freq) * 100
	p2.ChangeDutyCycle(dc2)
	
def counterClockWise_L():
	global freq
	

	pW = 0.0016
	dc = (pW * freq) * 100
	p.ChangeDutyCycle(dc)
	
def counterClockWise_R():
	global freq
	
	pW2 = 0.0016
	dc2 = (pW2 * freq) * 100
	p2.ChangeDutyCycle(dc2)


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



def makeCenter(x):
	centerX = 320
	pivotStart = time.clock()
	
	
	#Determine Direction to move
	if x < centerX:
		print 'first'
		#Move left
		#pivot for 0.1 seconds
		while (time.clock() - pivotStart) < 0.1:
			pivotLeft()
		immediateStop()	
		
		
		
	elif x > centerX:
		print 'second'
		#Move right
		#pivot for 0.1 seconds
		while (time.clock() - pivotStart) < 0.1:
			pivotRight()	
		immediateStop()


################# END MOTOR STUFF ###################

			
			

	
	
	
	
	
	
				
			


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
	firstGray = grayOld
	grayNew = grayOld
	frameDiff = cv2.absdiff(grayNew, grayOld)
	threshOld = cv2.threshold(frameDiff, 25, 255, cv2.THRESH_BINARY)[1]
	
	foundObject = 0
	newNorm = 1
	drawNow = 0
	ready = 0
	
	try:
		for frame in camera.capture_continuous(rawCapture, format="bgr", 
		use_video_port=True):
			
			frame =frame.array
		
			frame = frame.reshape((480,640,3))
		
			grayNew = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
			frameDiff = cv2.absdiff(grayNew, grayOld)
			thresh = cv2.threshold(frameDiff, 50, 255, cv2.THRESH_BINARY)[1]
		
			
			if not foundObject:
				
				zeroNorm = abs(norm(thresh) - norm(threshOld))
				
		
			
			if zeroNorm and newNorm:
				
				foundObject = 1
				#thresh = cv2.threshold(frameDiff, 100, 255, cv2.THRESH_BINARY)[1]
				newNorm = abs(norm(thresh) - norm(threshOld))
			
			#Draw when image stops changing
			if not newNorm and not drawNow and not ready:
				assert newNorm == 0.0
				
				
				#When it stops changing compare with first frame
				changeDiff = cv2.absdiff(grayNew, firstGray)
				threshChange = cv2.threshold(changeDiff, 50, 255, cv2.THRESH_BINARY)[1]
				cannyChange =cv2.Canny(threshChange, 50, 100)
			
				changeContours, _ = cv2.findContours(cannyChange, cv2.RETR_TREE, 
				cv2.CHAIN_APPROX_SIMPLE)
				
				
				cv2.drawContours(frame, changeContours, -1, (0,0,255), 2)
				
				
				#Calculate Centroid
				try:
					maxC = max(changeContours, key=cv2.contourArea)
					M = cv2.moments(maxC)
					cX, cY =(int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
				
					#drawNow = 1	
				except ValueError:
					continue
				
			
			#if drawNow:
				cv2.drawContours(frame, changeContours, -1, (0,0,255), 2)
				
				#Move Object to Make cX equal to its Center
				if (cX < 290) or (cX > 350): #and not ready:
					makeCenter(cX)
					print 'not ready'
					
				else:
					ready = 1
					print 'ready'
					immediateStop()
				
				
				
				
				print cX
	
			
			
			
					
				
		
			cv2.imshow("Tracking", frame)
			
			oldFrame = frame
			grayOld =  cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)
			
			threshOld = thresh
			
			
			
			
			key = cv2.waitKey(60) & 0xFF
			
			
			#Clear current frame for next frame
			rawCapture.truncate(0)
		
			if key == ord("q"):
				print 'quitting'
				break
			
	except KeyboardInterrupt:
		
		rawCapture.truncate(0)
		print 'here'
		GPIO.cleanup()	
		
		cv2.destroyAllWindows()


	cv2.destroyAllWindows()
	GPIO.cleanup()




tracker()






