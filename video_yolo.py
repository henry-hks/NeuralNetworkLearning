import cv2
import time

import sys
sys.path.append("/home/henry-etech/darknet")
import darknet as dn
import darknet_images as dni

#load yolov4
#tiny
# net, class_names, class_colors = dn.load_network("/home/henry-etech/darknet/cfg/yolov4-tiny.cfg","/home/henry-etech/darknet/cfg/coco.data","/home/henry-etech/darknet/cfg/yolov4-tiny.weights")

#full
net, class_names, class_colors = dn.load_network("/home/henry-etech/darknet/cfg/yolov4.cfg","/home/henry-etech/darknet/cfg/coco.data","/home/henry-etech/darknet/cfg/yolov4.weights")

#load video
# videoPath = '/home/henry-etech/autonomous_driving/traffic_video.mp4'
# videoPath = '/home/henry-etech/autonomous_driving/Dhaka_street_view.mp4'
videoPath = '/home/henry-etech/autonomous_driving/traffic_hk.mp4'
video = cv2.VideoCapture(videoPath)
start_time = time.time()
frame_id = 0

while True:
    ret, frame = video.read()
    frame_id += 1

    if not ret:
        video = cv2.VideoCapture(videoPath)
        continue
    
    detected_frame, detection = dni.frame_detection(frame, net, class_names, class_colors, 0.8)

    end_time = time.time()
    fps = round(frame_id/(end_time - start_time), 3)
    cv2.putText(detected_frame, "FPS: {}".format(fps), (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.imshow("frame", detected_frame)

    key = cv2.waitKey(25)
    if key == 27:
        cv2.imwrite("result.jpg", detected_frame)
        break

video.release()
cv2.destroyAllWindows()
