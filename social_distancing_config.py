# base path to YOLO directory
# this is used to direct the main function to model path 
MODEL_PATH = "yolo-coco"

# initialize minimum probability to filter weak detections... that is 30% is minimum 
#when an object's confidence to the guessed thing is less than 30% it will be marked as other object
MIN_CONF = 0.3
# the threshold when applying non-maxima suppression
#non maxima suppression,if an image is recognised as more number of small images or when an imaage is recognised in more than one angle or way to choose the correct object this is used
#crt explanation is when overlapping of two or more object ,then this value is min...that is this helos to identify more number of individual overlapped
  
NMS_THRESH = 0.3

# boolean indicating if NVIDIA CUDA GPU should be used
USE_GPU = True

# define the minimum safe distance (in pixels) that two people can be
# from each other
MIN_DISTANCE = 50








