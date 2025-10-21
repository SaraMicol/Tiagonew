#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import time

class PersistentObjects:
    def __init__(self):
        # --- Subscribers ---
        rospy.Subscriber("/pcl_objects", PointCloud2, self.cb_centroids)

        # --- Publishers ---
        self.pub_centroids = rospy.Publisher("/pcl_persistent", PointCloud2, queue_size=10, latch=True) 
         # Usando PointCloud2 invece di PointCloud1 perchè dava problemi

        # --- Internal storage ---
        self.objects = []  # List of centroid points (Point)
        
        rospy.loginfo("Persistent object daemon started.")

    # -------------------- CALLBACKS --------------------
    def cb_centroids(self, msg):
        """Callback to update centroids."""
        # tutti i punti che formano pointcloud
        point_list = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)

        # Aggiungi i nuovi punti alla lista persistente, evitando duplicati
        for point in point_list:
            if point not in self.objects:
                self.objects.append(point)

        self.publish_persistent()

    # -------------------- PUBLISH --------------------
    def publish_persistent(self):
        """Publish the points persistently as PointCloud2."""
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "xtion_rgb_optical_frame"

        # Converting the PointCloud (list of Points) to PointCloud2 format
        point_cloud_data = pc2.create_cloud_xyz32(header, self.objects)

        # Publish the PointCloud2
        self.pub_centroids.publish(point_cloud_data)

        rospy.loginfo(f"Published {len(self.objects)} centroids as PointCloud2.")

# -------------------- MAIN --------------------
if __name__ == "__main__":
    rospy.init_node("persistent_objects_daemon")
    node = PersistentObjects()
    rospy.spin()
