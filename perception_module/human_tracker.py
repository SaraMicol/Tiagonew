#!/usr/bin/env python3
"""
Nodo ROS che usa MediaPipe per tracciare umani.
Pubblica:
- `/human_landmarks` (visualization_msgs/MarkerArray) contenente sfere/linee per ogni landmark
- `/human_tracker/image` (sensor_msgs/Image) immagine annotata RGB per debugging

"""

import os
import rospy
from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray
from cv_bridge import CvBridge
import numpy as np
from cv_utils import  points_list_to_rviz
import utils
import cv2
import mediapipe as mp
import time



class HumanTracker:

    def __init__(self):
        self.pub_markers = rospy.Publisher('/human_landmarks', MarkerArray, queue_size=1, latch=True)
        self.pub_image = rospy.Publisher('/human_keypoints', Image, queue_size=1, latch=True)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                       min_detection_confidence=0.5,
                                       min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        self.bridge = CvBridge()

    def publish_landmarks(self, results, image, depth, camera_info):
        right_hand_idx = [15, 17, 19, 21]
        left_hand_idx = [16, 18, 20, 22]
        right_hand_points = []
        left_hand_points = []
        body_points = []
        h, w, _ = image.shape
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                x_pixel = landmark.x * w
                y_pixel = landmark.y * h
                if idx in right_hand_idx:
                    right_hand_points.append((x_pixel, y_pixel))
                elif idx in left_hand_idx:
                    left_hand_points.append((x_pixel, y_pixel))
                else:
                    body_points.append((x_pixel, y_pixel))

            # Pubblica i marker per ogni gruppo
            if right_hand_points:
                points_list_to_rviz(
                    right_hand_points,
                    depth,
                    camera_info,
                    color = "green",
                    frame_id=camera_info.header.frame_id,
                    topic="/human_landmarks_right_hand",
                    marker_scale=0.07,
                )
            if left_hand_points:
                points_list_to_rviz(
                    left_hand_points,
                    depth,
                    camera_info,
                    color = "magenta",
                    frame_id=camera_info.header.frame_id,
                    topic="/human_landmarks_left_hand",
                    marker_scale=0.07,
                )
            if body_points:
                points_list_to_rviz(
                    body_points,
                    depth,
                    camera_info,
                    color = "blue",
                    frame_id=camera_info.header.frame_id,
                    topic="/human_landmarks_body",
                    marker_scale=0.05,
                )

    def track_human(self):
        synced = utils.get_synchronized_frame(timeout=0.1)
        if synced is None:
            return
        image, depth, camera_info = synced
        
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        start_time = time.time()
        results = self.pose.process(img_rgb)
        self.publish_landmarks(results, image, depth, camera_info)
        if results.pose_landmarks:
            img_with_landmarks = img_rgb.copy()
            self.mp_drawing.draw_landmarks(img_with_landmarks, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            img_with_landmarks = cv2.cvtColor(img_with_landmarks, cv2.COLOR_RGB2BGR)
            path_file = os.path.dirname(os.path.abspath(__file__))
            #save image in ../assets/debug_human_detections.jpg
            cv2.imwrite(os.path.join(path_file, "../assets/debug_human_detections.jpg"), img_with_landmarks)
            img_msg = self.bridge.cv2_to_imgmsg(img_with_landmarks, encoding="bgr8")
            self.pub_image.publish(img_msg)
        #rospy.loginfo("Time for human tracking: %f", time.time() - start_time)


if __name__ == "__main__":
    rospy.init_node("human_tracker")
    node = HumanTracker()
    while not rospy.is_shutdown():
        node.track_human()
