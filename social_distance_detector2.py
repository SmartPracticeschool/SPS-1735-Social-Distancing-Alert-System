#this file is used to load the videowhich we are going to use in detection file
#and also we are loading yolo,weight
#weight eg--to detect an horse we may see many things along with the horse like tree,land....extra these are called weights
#this is front end since we are classifing camera
#this gives the results like confidence,probability,b box these were rturned in yolo file
from packages import social_distancing_config as config
from packages.detection import detect_people
#this is used for array...like finding nearest distance.... 
from scipy.spatial import distance as dist
import numpy as np
#imutils is used to detect the picture by rotating,capturing in different angle
import imutils
import cv2
#our os
import os
#yolo is the only algorithm that is used to detect and classification

# load the COCO class labels our YOLO model was trained on

labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])
# load our YOLO object detector trained on COCO dataset (80 classes)

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if config.USE_GPU:
    # set CUDA as the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
#now only video processing starts

vs = cv2.VideoCapture(r"pedestrians.mp3"if "pedestrians.mp3"else 0)
writer = None


# loop over the frames from the video stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break
    # resize the frame and then detect people (and only people) in it
    frame = imutils.resize(frame, width=700)
    results = detect_people(frame, net, ln,personIdx=LABELS.index("person"))
    # initialize the set of indexes that violate the minimum social
    # distance
    violate = set()

    # ensure there are *at least* two people detections (required in
    # order to compute our pairwise distance maps)
    #if the number of persons in the image is greater than 2 then need to check violation
    # so this file only determines the violation
    if len(results) >= 2:
        # extract all centroids from the results and compute the
        # Euclidean distances between all pairs of the centroids
        centroids = np.array([r[2] for r in results])
        #centroid hold the centroid value of 2 person 
        ##dont thing that 2 parameters are centroid the algorithm will find the difference between the two centroids
        D = dist.cdist(centroids, centroids, metric="euclidean")

        # loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                # check to see if the distance between any two
                # centroid pairs is less than the configured number
                # of pixels
                if D[i, j] < config.MIN_DISTANCE:
                    # update our violation set with the indexes of
                    # the centroid pairs
                    violate.add(i)
                    violate.add(j)

    # loop over the results
    #now we need to set the bounding box for that we are using loop in which we sends the complete details of the person that is result
    for (i, (prob, bbox, centroid)) in enumerate(results):
        # extract the bounding box and centroid coordinates, then
        # initialize the color of the annotation
        (startX, startY, endX, endY) = bbox#here this is bounding box with h,w,l,d
        (cX, cY) = centroid#this is the centroid of the person
        color = (0, 255, 0)#here generally for all person the color is green

        # if the index pair exists within the violation set, then
        # update the color
        if i in violate:#if person i in voiate state then change it red color
            color = (0, 0, 255)
        # draw (1) a bounding box around the person and (2) the
        # centroid coordinates of the person,
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)
    # draw the total number of social distancing violations on the
    # output frame
    text = "Social Distancing Violations: {}".format(len(violate))
    cv2.putText(frame, text, (10, frame.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
    

    # check to see if the output frame should be displayed to our
    # screen
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break    
    # if an output video file path has been supplied and the video
    # writer has not been initialized, do so now
    if r"social-distance-detector" != "" and writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(r"output.avi", fourcc, 25,
            (frame.shape[1], frame.shape[0]), True)
    # if the video writer is not None, write the frame to the output
    # video file
    if writer is not None:
        writer.write(frame)       
cv2.destroyAllWindows()


