# import the necessary packages
#this is backend since we are analysing the frames
from .social_distancing_config import NMS_THRESH
from .social_distancing_config import MIN_CONF
import numpy as np
#opencv to capture the video or image
import cv2
#in this fn we are getting the frame, nerwork(neural network),output and person index
##frame: image frame from the webcam/ your video file
##net: pre-trained YOLO weights model
##ln: Yolo Output model layers
##personIdx: Yolo algorithm is trained on so many objects from that we are considering people class, the index specified is especially for people class
##(H, W) = grabs the dimensions of the frame to rescale the image
def detect_people(frame, net, ln, personIdx=0):
	# grab the dimensions of the frame and  initialize the list of
	# results
	(H, W) = frame.shape[:2]#here 3 parameters will be actual array so limiting to 2
	results = []
    ##results =[] :, the Yolo algorithms give 3 parameters as output which is stored in the results list variable
##probability of detected class
##bounding box coordinates
##Centroid of the object
    
    

	# construct a blob from the input frame and then perform aforward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
    ##blob is used to convert the thins in the frame to algorithm understandable manner
    ##dnn=deep neural network
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)#to no more about these parameter refer https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
	net.setInput(blob)#this will set the processed imafge or frame to the network that is neural network
	layerOutputs = net.forward(ln)#in this the information is send to the fianl layer which is neural network
##the final things which we get from the neural network is stored in layero
	# initialize our lists of detected bounding boxes, centroids, and
	# confidences, respectively
	boxes = []
	centroids = []
	confidences = []
	for output in layerOutputs:
		for detection in output:
#detection have 8 values they are confidence,x,y,h,w,pr(1),pr(2),pr(3) and so on
            #scores pr1 ,pr2 are classid 
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if classID == personIdx and confidence > MIN_CONF:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height)=box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes,confidences,MIN_CONF,NMS_THRESH)
	if len(idxs) > 0:
        #this loop run for each people in a frame
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# update our results list to consist of the person
			# prediction probability, bounding box coordinates,
			# and the centroid
			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(r)

	# return the list of results
	return results
