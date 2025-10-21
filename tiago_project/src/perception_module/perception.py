#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from cv_utils import *
import utils
import os
import tf2_ros
from cv_bridge import CvBridge
import numpy as np
import cv2
from models import DINO, VitSam, OWLv2
from dataclasses import dataclass
from typing import Tuple
from cv_utils import _transform_point_xyz
from segmentator import obtain_centroids
from matplotlib.colors import to_rgb
import time

from tiago_project.msg import Centroid, CentroidArray

@dataclass
class Detection:
    bbox: Tuple[float, float, float, float] 
    label: str
    score: float
    mask: np.ndarray

class DetectObjects:
    def __init__(self):
        self.pub_image = rospy.Publisher("/image_with_bb", Image, queue_size=10, latch=True)
        self.publish_objects_names = rospy.Publisher("/detected_object_names", String, queue_size=10, latch=True)
        
        # Publisher for current object markers (green bounding boxes)
        self.current_markers_pub = rospy.Publisher('/current_object_markers', MarkerArray, queue_size=1, latch=True)

        self.COLORS = ['red', 'green', 'blue', 'magenta', 'gray', 'yellow'] * 3

        self.owlv2 = OWLv2()
        #self.owlv2 = DINO()
        self.vitsam = VitSam(utils.ENCODER_VITSAM_PATH, utils.DECODER_VITSAM_PATH)

        self.centroid_pub=rospy.Publisher("/centroids_custom", CentroidArray, queue_size=1, latch=True)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener= tf2_ros.TrasformListener(self.tf.buffer)

        # Log device information so it's easy to verify GPU usage at runtime
        try:
            owl_dev = getattr(self.owlv2, 'device', 'cpu')
            vs_dev = getattr(self.vitsam, 'device', 'cpu')
            rospy.loginfo(f"OWLv2 device: {owl_dev}")
            rospy.loginfo(f"VitSam device: {vs_dev}")
        except Exception:
            pass

        # If CUDA is available, enable cuDNN benchmark for potential perf gains
        try:
            import torch
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                rospy.loginfo("CUDA available: enabled cuDNN benchmark")
        except Exception:
            pass

        self.bridge = CvBridge()
        
        # Store filtered object names for visualization
        self.filtered_objects = []
        
        # Subscribe to filtered object visualization data
        rospy.Subscriber("/filtered_object_viz", String, self._filtered_viz_callback)

    def _filtered_viz_callback(self, msg: String):
        """Callback for filtered object visualization data."""
        if msg.data:
            self.filtered_objects = [name.strip() for name in msg.data.split(',') if name.strip()]
        else:
            self.filtered_objects = []
    

   
    def color_pcl(self, image, detections):
        """
        Visualizza le maschere e i centroidi usando le
        utility in `cv_utils.py`.
        Per ogni detection:
        - prepara la maschera 2D
        - calcola il centro della maschera, riproietta quel singolo punto in 3D
        usando la depth e `camera_info` e pubblica una sfera con `point_to_rviz`.
        """
        # Ottieni camera_info e depth_array direttamente come numpy array
        camera_info = utils.get_latest_camera_info(timeout=1.0)
        depth_array = utils.get_latest_depth_array(timeout=1.0)
        
        if camera_info is None or depth_array is None:
            rospy.logwarn_throttle(5.0, 'color_pcl: missing depth or camera_info, skipping pointcloud publication')
            return
        
        # depth_array � gi� un numpy array in metri, pronto all'uso
        depth_img = depth_array
        
        # Applica overlay delle maschere sull'immagine e costruisci una lista di mask2d
        mask_list = []
        for idx, detection in enumerate(detections):
            mask = detection.mask
            mask2d = mask[:, :, 0]
            mask2d = (mask2d.astype(np.uint8) * 255)
            overlay_mask_on_image(image, mask2d, color_rgb=to_rgb(self.COLORS[idx % len(self.COLORS)]), alpha=0.5)
            mask_list.append(mask2d)
        
        # Pubblica un unico PointCloud2 aggregato con object_id per ogni maschera
        try:
            labels = [det.label if hasattr(det, 'label') else f"obj_{i}" for i, det in enumerate(detections)]
            string_labels = ','.join(labels)
            self.publish_objects_names.publish(String(data=string_labels))
            
            mask_list_to_pointcloud2(
                mask_list, 
                depth_img, 
                camera_info, 
                labels=labels, 
                frame_id=camera_info.header.frame_id, 
                topic="/pcl_objects",
                max_points_per_obj=1500,
            )
            
            # Pubblica PointCloud2 individuali per ogni oggetto automaticamente
            try:
                publish_individual_pointclouds_by_id(
                    mask_list, 
                    depth_img, 
                    camera_info, 
                    labels=labels, 
                    frame_id=camera_info.header.frame_id, 
                    topic_prefix="/pcl_id",
                )
            except Exception as e:
                rospy.logwarn(f"Errore publish_individual_pointclouds_by_id: {e}")
                
        except Exception as e:
            rospy.logwarn(f"Errore publishing aggregated PointCloud2: {e}")
        
        return
    def run_detection(self, image, depth):
        print("Running detection...")
        # DINO labels all tools as "tool"
        #labels = ["brown_table", "screwdriver", "tool", "pincers", "wrench", "hammer"]
        labels = ["table", "screwdriver", "pinchers", "wrench","hammer"]
        self.owlv2.set_classes(labels)

        
        # Time the OWLv2 prediction (text+image model)
        t_predict_start = time.time()
        overall_start = time.time()
        bboxs, labels, scores = self.owlv2.predict(image, box_threshold=0.3, text_threshold=0.25)
        t_predict = time.time() - t_predict_start

        detections = []
        # normalize duplicate label names to keep them unique
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        new_labels = []
        label_indices = {}
        for label in labels:
            if label_counts[label] > 1:
                if label not in label_indices:
                    label_indices[label] = 1
                new_label = f"{label}_{label_indices[label]}"
                label_indices[label] += 1
                new_labels.append(new_label)
            else:
                new_labels.append(label)
        labels = new_labels

        # Time mask generation (VitSam) per box
        mask_times = []
        total_masks = 0
        for bbox, label_name, score in zip(bboxs, labels, scores):
            t_mask_start = time.time()
            masks, _ = self.vitsam(image.copy(), bbox)
            t_mask = time.time() - t_mask_start
            # record timing and count
            mask_times.append(t_mask)
            total_masks += len(masks) if hasattr(masks, '__len__') else 1

            for mask in masks:
                mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask)
                mask_np = mask_np.transpose(1, 2, 0)
                detection = Detection(
                    bbox=tuple(bbox),
                    label=label_name,
                    score=score,
                    mask=mask_np
                )
                detections.append(detection)

        # Time color_pcl (mask->pointcloud publishing)
        t_color_start = time.time()
        self.color_pcl(image, detections)
        t_color = time.time() - t_color_start

        overall_time = time.time() - overall_start

        # Compute mask stats
        mask_total = sum(mask_times) if mask_times else 0.0
        mask_avg = mask_total / len(mask_times) if mask_times else 0.0

        rospy.loginfo(
            "run_detection timings: predict=%.3fs, mask_calls=%d, mask_total=%.3fs, mask_avg=%.3fs, color_pcl=%.3fs, overall=%.3fs",
            t_predict,
            len(mask_times),
            mask_total,
            mask_avg,
            t_color,
            overall_time,
        )

        return detections

    def _create_current_object_markers(self, detections, depth_image, camera_info, frame_id="camera_link"):
        """Create markers for current object detections (green spheres)."""
        marker_array = MarkerArray()
        
        # Get camera intrinsic parameters
        K = camera_info.K
        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]
        
        for marker_id, detection in enumerate(detections):
            # Get center of bounding box
            x1, y1, x2, y2 = detection.bbox
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Check bounds
            if center_x < 0 or center_y < 0 or center_y >= depth_image.shape[0] or center_x >= depth_image.shape[1]:
                continue
                
            # Get depth at center
            z = float(depth_image[center_y, center_x])
            if z <= 0 or np.isnan(z):
                continue
                
            # Convert to 3D coordinates
            X = (center_x - cx) * z / fx
            Y = (center_y - cy) * z / fy
            
            # Create marker
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = "current_objects"
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(X)
            marker.pose.position.y = float(Y)
            marker.pose.position.z = float(z)
            marker.pose.orientation.w = 1.0
            
            # Set scale
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            
            # Set green color for current objects
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)  # Green
            marker.lifetime = rospy.Duration(1.0)  # Short lifetime for current detections
            
            marker_array.markers.append(marker)
        
        return marker_array

    def publish_objects(self):
        
        # -Ottieni frame sincronizzati ---
        synced = utils.get_synchronized_frame(timeout=1)
        if synced is None:
            rospy.logwarn("No synchronized frame available yet.")
            return
        image, depth, camera_info = synced

        # - Esegui rilevamento oggetti ---
        detections = self.run_detection(image, depth)
        mask_list = [det.mask[:, :, 0] for det in detections]  # lista di maschere 2D

        # -- Pubblica PointCloud degli oggetti ---
        self.color_pcl(image, detections)  # gestisce sia pcl aggregata che individuali

        # - Calcola centroidi 3D da maschere ---
        centroids_3d = mask_list_to_centroid(mask_list, depth, camera_info)

        # -- Pubblica CentroidArray ---
        from tiago_project.msg import Centroid, CentroidArray
        centroid_array_msg = CentroidArray()
        centroid_array_msg.header.stamp = rospy.Time.now()
        centroid_array_msg.header.frame_id = camera_info.header.frame_id

        points_for_rviz = []
        labels_for_rviz = []
        for det, c in zip(detections, centroids_3d):
            if c is None:
                continue
            # Crea messaggio Centroid
            centroid_msg = Centroid()
            centroid_msg.x, centroid_msg.y, centroid_msg.z = c
            centroid_msg.label = det.label
            centroid_array_msg.centroids.append(centroid_msg)

            # Prepara lista per RViz
            points_for_rviz.append(c)
            labels_for_rviz.append(det.label)

        self.centroid_pub.publish(centroid_array_msg)

        #  Visualizza centroidi in RViz ---
        points_list_to_rviz_3d(
            points_for_rviz,
            labels=labels_for_rviz,
            frame_id=camera_info.header.frame_id,
            topic="/centroid_markers",
            marker_scale=0.06
        )

        # --- Step 7: Disegna bounding box sull’immagine e pubblica ---
        img_with_bb = draw_detections(image, detections)
        img_msg = self.bridge.cv2_to_imgmsg(img_with_bb, encoding="bgr8")
        self.pub_image.publish(img_msg)

        rospy.loginfo("publish_objects: completato ciclo rilevamento e pubblicazioni")

if __name__ == "__main__":
    rospy.init_node("detection_node")
    node = DetectObjects()
    rate = rospy.Rate(1/20.0)  # 1 cycle every 30 seconds
    while not rospy.is_shutdown():
        node.publish_objects()