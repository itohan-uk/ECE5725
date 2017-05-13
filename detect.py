from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
from scipy.linalg import norm
import RPi.GPIO as GPIO
import os
import subprocess

#subprocess.call("fbcp &", shell=True)
#subprocess.call("raspivid -t 0", shell=True)

os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.putenv('SDL_FBDEV'      , '/dev/fb1')
os.putenv('SDL_MOUSEDRV'   , 'TSLIB')
os.putenv('SDL_MOUSEDEV'   , '/dev/input/touchscreen')


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

################# SENSOR STUFF ###################
distance =0
stop =0 #set stop low for object finding
TRIG=19
ECHO=26
#needed for ultrasonic sensor#
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

GPIO.output(TRIG, False)


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

'''
def resume():
	global dc
	global dc2

	p.ChangeDutyCycle(dc)
	p2.ChangeDutyCycle(dc2)
'''



def backward():
	#global firstTime
	#firstTime = time.clock()
	clockWise_R()
	counterClockWise_L()
	


def forward():
	#print "forward"
	clockWise_L()
	counterClockWise_R()
	
	


def pivotLeft():
	clockWise_R()
	clockWise_L()

def pivotRight():
	counterClockWise_R()
	counterClockWise_L()









def reCenter(frame):
	
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		hue, saturation, value = cv2.split(hsv)
		ret, thresh = cv2.threshold(saturation, 60, 255, cv2.THRESH_BINARY +
		 cv2.THRESH_OTSU)
		
		medianFilter = cv2.medianBlur(thresh, 7)
		
		contours, _ = cv2.findContours(medianFilter, cv2.RETR_TREE, 
				cv2.CHAIN_APPROX_SIMPLE)
		
		#Calculate Centroid
		try:
			maxC = max(contours, key=cv2.contourArea)
			area = cv2.contourArea(maxC)
			
			M = cv2.moments(maxC)
			cX, cY =(int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
			
			
		except ValueError:
			pass
		
		except ZeroDivisionError:
			pass
		

		if area < 18000:
			(xStart, xEnd) = (int(cX) - 30), (int(cX) + 30)
			(yStart, yEnd) = (int(cY) - 30), (int(cY) + 30)
			cv2.line(frame, (xStart, cY), (xEnd, cY), (0, 0, 255), 3)
			cv2.line(frame, (cX, yStart), (cX, yEnd), (0, 0, 255), 3)
			makeCenter(cX)




def makeCenter(x):
	centerX = 320
	pivotStart = time.clock()
	
	
	#Determine Direction to move
	if x < centerX:
		#pivot left for 0.05 seconds
		while (time.clock() - pivotStart) < 0.05:
			pivotLeft()
			#pivotRight()	
			#print "left"
	
		
		
	elif x > centerX:
		#pivot right for 0.05 seconds
		while (time.clock() - pivotStart) < 0.05:
			pivotRight()
			#print 'right' 
			#pivotLeft()	



def picker(x ,y, w, h, grayNew, firstGray):
	#global grayNew
	#global firstGray
	forwardStart = time.clock()
	#while x > 50 and y > 50 and (x + w) < 550 and (y + h) < 420:
	if 	not (x > 50 and y > 50 and (x + w) < 550 and (y + h) < 420):
		forward()
		
	else:
		immediateStop()
		
				
################# END MOTOR STUFF ###################	

def distance_sense():
	global distance
	
	print "waiting for sensor to settle"
	time.sleep(.1)

	GPIO.output(TRIG, True)
	time.sleep(.00001)
	GPIO.output(TRIG, False)

	while GPIO.input(ECHO)==0:
		pulse_start = time.time()
	
	while GPIO.input(ECHO)==1:
		pulse_end = time.time()

	pulse_duration = pulse_end - pulse_start

	distance = pulse_duration * 17150

	distance = round(distance, 2)

	print "Distance: ",distance,"cm"
	
############### TRACKER PART BEGINS #################	
				
			


def tracker():
	camera = PiCamera()
	camera.rotation = 180
	#camera.resolution = (320,240)
	camera.resolution = (640,480)
	camera.framerate = 32
	camera.brightness = 60

	#rawCapture = PiRGBArray(camera ,size=(320,240))
	rawCapture = PiRGBArray(camera ,size=(640,480))

	time.sleep(0.1)
	#frame = np.empty((240 * 320 * 3), dtype=np.uint8)
	frame = np.empty((480 * 640 * 3), dtype=np.uint8)
	
	camera.capture(frame, format="bgr")
	#frame = frame.reshape((240,320,3))
	frame = frame.reshape((480,640,3))


	
	oldFrame = frame
	grayOld =  cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)
	firstGray = grayOld
	grayNew = grayOld
	frameDiff = cv2.absdiff(grayNew, grayOld)
	threshOld = cv2.threshold(frameDiff, 80, 255, cv2.THRESH_BINARY)[1]
	
	foundObject = 0
	newNorm = 1
	#drawNow = 0
	ready = 0
	forwardStatus = 0
	startTime = 0
	
	#for sensor#
	global stop
	try:
		for frame in camera.capture_continuous(rawCapture, format="bgr", 
		use_video_port=True):
			
		
			
			
			frame =frame.array
		
			#frame = frame.reshape((240,320,3))
			frame = frame.reshape((480,640,3))
		
			grayNew = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			
		
			#Get the absolute difference between the first frame and this frame
			frameDiff = cv2.absdiff(grayNew, grayOld)
			
			#Get the threshold of the absolute difference
			thresh = cv2.threshold(frameDiff, 80, 255, cv2.THRESH_BINARY)[1]
			
			
			if not foundObject:
				
				#Compare the old threshold and the new threshold
				zeroNorm = abs(norm(thresh) - norm(threshOld))
				#print 'not found'
				#print zeroNorm
		
			#If a change was detected, compute the norm until
			#no change is detected
			if zeroNorm and newNorm:
				#print 'found'
				foundObject = 1
				thresh = cv2.threshold(frameDiff, 100, 255, cv2.THRESH_BINARY)[1]
				newNorm = abs(norm(thresh) - norm(threshOld))
				
			#Draw when image stops changing
			#if not newNorm and not drawNow and not ready:
			#print 'cake'
			#print 'new %f'%newNorm
			#print 'zero %f'%zeroNorm
			#print newNorm
			if not newNorm and not ready:
				assert newNorm == 0.0
				
				
				
				
				#When it stops changing compare with first frame
				changeDiff = cv2.absdiff(grayNew, firstGray)
				threshChange = cv2.threshold(changeDiff, 50, 255, cv2.THRESH_BINARY)[1]
				cannyChange =cv2.Canny(threshChange, 50, 100)
			
				changeContours, _ = cv2.findContours(cannyChange, cv2.RETR_TREE, 
				cv2.CHAIN_APPROX_SIMPLE)
				
				
				#cv2.drawContours(frame, changeContours, -1, (0,0,255), 2)
				
				
				#Calculate Centroid
				try:
					maxC = max(changeContours, key=cv2.contourArea)
					M = cv2.moments(maxC)
					cX, cY =(int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
					
					
					#drawNow = 1	
				except ValueError:
					continue
				
				except ZeroDivisionError:
					continue
				
			
				#cv2.drawContours(frame, changeContours, -1, (0,0,255), 2)
					
				#Move Object to Make cX equal to its Center
				if (cX < 290) or (cX > 350): #and not ready:
				#if (cX < 130) or (cX > 200): #and not ready:
			
					print cX
		
					#makeCenter(cX) #Centralize the object
					reCenter(frame)
					immediateStop()	
					print 'not ready'
					
				else:
					ready = 1
					print 'ready'
					immediateStop()
				
			
				
			if ready:
				#Draw cross bar using cX and cY
				#print ready
				
				
				(xStart, xEnd) = (int(cX) - 30), (int(cX) + 30)
				(yStart, yEnd) = (int(cY) - 30), (int(cY) + 30)
				cv2.line(frame, (xStart, cY), (xEnd, cY), (0, 0, 255), 3)
				cv2.line(frame, (cX, yStart), (cX, yEnd), (0, 0, 255), 3)
				(x ,y, w, h) = cv2.boundingRect(maxC)
				#cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
				
				
				time.sleep(0.1)
				#forward()

				
				
				#check distance between object and robot
				distance_sense()
				if (distance <7 and stop==0):
					print "object in path"
					immediateStop()
					stop =1
					
				elif (stop!=1):
					forward()
					
				########end added stuff from Anthony#####333
			
			#Move forward for a particular time, stop and recentralize
			if not forwardStatus and ready and stop:
				startTime = time.clock()
				forwardStatus  = 1
				#stop  = 0
			if (time.clock() - startTime) >= 0.5:
				ready = 0
				#reCenter(frame)
				forwardStatus  = 0
				
			print 'stop %d'%stop
			
			################## CHECK IF TIME TO PICK #######################
			#Check if  its ready to pick up the object
			#if 	(x <= 50 and y <= 50 and (x + w) >= 600 and (y + h) >= 440):	
			#	immediateStop()
			#	print 'yay!!! found' 
				
				
				
				#TO DO
				#pICK oBJECT
				
				#TO DO
				#Move backward for sometime
				#Rotate around slowly looking for a particular object
				#When you find it, move towards it, drop the object
				
				#move backward
				#Reset all the variables and begin again
				
				
			###################################################################	
				              
				
			
				
				
			
				
				
				
				
				#picker(x ,y, w, h, grayNew, firstGray)
				#changeDiff = cv2.absdiff(grayNew, firstGray)
				#threshChange = cv2.threshold(changeDiff, 50, 255, cv2.THRESH_BINARY)[1]
				#cannyChange =cv2.Canny(threshChange, 50, 100)
			
				#changeContours, _ = cv2.findContours(cannyChange, cv2.RETR_TREE, 
				#cv2.CHAIN_APPROX_SIMPLE)
				
				#try:
				#	maxC = max(changeContours, key=cv2.contourArea)
					#M = cv2.moments(maxC)
					#cX, cY =(int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
				
					#drawNow = 1	
				#except ValueError:
				#	continue
				
				#picker(x ,y, w, h, grayNew, firstGray)
				#cv2.drawContours(frame, changeContours, -1, (0,0,255), 2)	
			
				
		
			cv2.imshow("Tracking", frame)
			
			oldFrame = frame
			grayOld =  cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)
			
			threshOld = thresh
			
			
			
			
			key = cv2.waitKey(60) & 0xFF
			
			
			#Clear current frame for next frame
			rawCapture.truncate(0)
			#print 'truncate'
			#rawCapture.seek(0)
			
		
		
			if key == ord("q"):
				print 'quitting'
				break
	
			
	except KeyboardInterrupt:
		print 'except'
		
		rawCapture.truncate(0)
		#rawCapture.seek(0)
		GPIO.cleanup()	
		cv2.destroyAllWindows()


	
	finally:
		
		rawCapture.truncate(0)
		#rawCapture.seek(0)
		cv2.destroyAllWindows()
		#camera.release()
		GPIO.cleanup()
		print 'finally'

##################### TRACKER PART ENDS #######################
		



tracker()

#while 1:
	#counterClockWise_L()
	#counterClockWise_R()
	#time.sleep(1)
	#clockWise_L()
	#clockWise_L()
#	forward()
#	print 'here'
#	pivotLeft()
	






