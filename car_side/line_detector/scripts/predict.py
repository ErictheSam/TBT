#!/usr/bin/env python3

from cv_bridge import CvBridgeError
from path_follower.msg import Lines
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

import cv2
import cv_bridge
import inference as inference_utils
import numpy as np
import os
import queue
import rospy
import threading
import multiprocessing as mp
import utils


class SomeProcess(threading.Thread):
    """Worker thread that publishes detected lines and intersections from inference outputs."""

    def __init__(self, queue, bridge):
        super().__init__()
        self.queue = queue
        self.bridge = bridge
        self.line_pub = rospy.Publisher("/output/lines_and_intersecs",Lines,queue_size=1)

    def run(self):
        msg_line = Lines()
        while not rospy.is_shutdown():
            batch_outputs, scores, indices = self.queue.get()
            pre_img = batch_outputs[0,:,:]
            displacement = batch_outputs[1:, :, :]
            segmentations, intersections = utils.pred_lines(
                pre_img, displacement, scores, indices,
                score_thr=0.03, dist_thr=40)
            
            pub_segments = [Float32MultiArray(data=line) for line in segmentations]
            pub_intersecs = [Float32MultiArray(data=pt) for pt in intersections]
            msg_line.lines = pub_segments
            msg_line.points = pub_intersecs

            try:
                self.line_pub.publish(msg_line)
            except CvBridgeError as e:
                print("error",e)

class Predictor:
    """Handles image preprocessing, TensorRT inference, and queues results for downstream line detection."""

    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.model_name = rospy.get_param("~model_name")
        self.image_size = rospy.get_param("~image_size")
        self.usbcam_height = rospy.get_param("~usbcam_height")
        self.usbcam_width = rospy.get_param("~usbcam_width")
        trt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "models", self.model_name)
        self.trt_inference_wrapper = inference_utils.TRTInference(trt_path, 1)
        self.image_sub = rospy.Subscriber(
            "/usb_cam/image_raw", Image, self.image_callback,
            queue_size=1, buff_size=2 ** 24
        )
        self.output_shapes = [(5, self.image_size, self.image_size),(200),(200)]
        self.msg_queue = queue.Queue(maxsize=1)
        self.bind_queue = mp.Queue(maxsize=1)
        self.clahe = cv2.createCLAHE(clipLimit=1.0,tileGridSize=(14,14))

    def image_callback(self, msg):
        """Callback for incoming ROS image messages: crops, preprocesses, and queues the frame."""
        if msg.header is not None:
            image_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            image_raw = image_raw[(self.usbcam_height - self.image_size) // 2:(self.usbcam_height + self.image_size) // 2,
                     (self.usbcam_width - self.image_size) // 2:(self.usbcam_width + self.image_size) // 2]

            image = cv2.medianBlur(image_raw,3)
            planes = cv2.split(image)
            gm = []
            for i in range(3):
                gm.append(self.clahe.apply(planes[i]))
            image = cv2.merge(gm)
            image = image.transpose((2,0,1))
            image = image / 255.0
            image = np.array(image, dtype=np.float32, order='C')
            self.msg_queue.put(image)

    def process_frame(self):
        """Grab the latest pre-processed image, run TensorRT inference, and queue the result."""
        image = self.msg_queue.get()
        predicted = self.trt_inference_wrapper.infer(image, self.output_shapes)
        self.bind_queue.put(predicted)

if __name__ == "__main__":

    rospy.init_node("inference")
    rospy.loginfo("Starting inference node")
    predictor = Predictor()

    sp = SomeProcess(predictor.bind_queue, predictor.bridge)
    sp.daemon = True
    sp.start()
    while not rospy.is_shutdown():
        predictor.process_frame()
