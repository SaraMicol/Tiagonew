#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from your_pkg_name.msg import CentroidArray  # 🔸 sostituisci con il tuo package!
import random
import numpy as np
import os


class ObjectManager:
    def __init__(self):
        # 🔹 Percorso della cartella in cui si trova questo script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.centroid_file_path = os.path.join(script_dir, "persistent_centroids.txt")

        self.last_centroids = {}  # label -> np.array([x,y,z])
        self.label_colors = {}    # per assegnare colori stabili
        self.frame_id = "map"

        # Subscriber e publisher
        self.centroid_sub = rospy.Subscriber(
            "/centroids_custom", CentroidArray, self._centroid_callback
        )
        self.marker_pub = rospy.Publisher(
            "/persistent_markers", MarkerArray, queue_size=1
        )

        # Servizio per cancellare i marker
        self.clear_srv = rospy.Service("/object_manager/clear_markers", Empty, self._clear_markers)

    # ==========================================================
    # Callback ricezione centroidi
    # ==========================================================
    def _centroid_callback(self, msg):
        """Aggiorna i centroidi persistenti e li salva su file"""
        for i, label in enumerate(msg.labels):
            centroid = np.array([msg.centroids[i].x, msg.centroids[i].y, msg.centroids[i].z])
            self.last_centroids[label] = centroid

        # Dopo aver aggiornato i centroidi, salva su file
        self._write_centroids_to_file()

        # E pubblica i marker persistenti
        self._publish_persistent_markers()

    # ==========================================================
    # Scrittura su file
    # ==========================================================
    def _write_centroids_to_file(self):
        """Scrive i centroidi persistenti in un file txt nella stessa cartella dello script"""
        try:
            with open(self.centroid_file_path, "w") as f:
                for label, centroid in self.last_centroids.items():
                    f.write(f"{label} {centroid[0]:.6f} {centroid[1]:.6f} {centroid[2]:.6f}\n")
            rospy.loginfo(f"[ObjectManager] Centroidi salvati su {self.centroid_file_path}")
        except Exception as e:
            rospy.logerr(f"[ObjectManager] Errore scrivendo i centroidi su file: {e}")

    # ==========================================================
    # Pubblica marker persistenti (sfere + label)
    # ==========================================================
    def _publish_persistent_markers(self):
        marker_array = MarkerArray()

        for i, (label, centroid) in enumerate(self.last_centroids.items()):
            x, y, z = centroid
            color = self._get_color_for_label(label)

            # --- Marker SFERA ---
            sphere = Marker()
            sphere.header.frame_id = self.frame_id
            sphere.header.stamp = rospy.Time.now()
            sphere.ns = "persistent_centroids"
            sphere.id = i
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = x
            sphere.pose.position.y = y
            sphere.pose.position.z = z
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.08
            sphere.color = color
            sphere.lifetime = rospy.Duration(0)
            marker_array.markers.append(sphere)

            # --- Marker TESTO ---
            text = Marker()
            text.header.frame_id = self.frame_id
            text.header.stamp = rospy.Time.now()
            text.ns = "persistent_labels"
            text.id = i + 1000
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = x
            text.pose.position.y = y
            text.pose.position.z = z + 0.12
            text.scale.z = 0.06
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = label
            text.lifetime = rospy.Duration(0)
            marker_array.markers.append(text)

        self.marker_pub.publish(marker_array)
        rospy.loginfo(f"[ObjectManager] Pubblicati {len(self.last_centroids)} marker persistenti")

    # ==========================================================
    # Colore casuale stabile per ogni label
    # ==========================================================
    def _get_color_for_label(self, label):
        if label not in self.label_colors:
            self.label_colors[label] = ColorRGBA(
                random.random(), random.random(), random.random(), 1.0
            )
        return self.label_colors[label]

    # ==========================================================
    # Servizio per cancellare tutti i marker
    # ==========================================================
    def _clear_markers(self, req):
        rospy.loginfo("[ObjectManager] Cancellazione di tutti i marker")
        self.last_centroids.clear()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array = MarkerArray()
        marker_array.markers.append(delete_all)
        self.marker_pub.publish(marker_array)

        # Cancella anche il file
        try:
            if os.path.exists(self.centroid_file_path):
                os.remove(self.centroid_file_path)
                rospy.loginfo(f"[ObjectManager] File {self.centroid_file_path} eliminato.")
        except Exception as e:
            rospy.logerr(f"[ObjectManager] Errore eliminando file centroidi: {e}")

        return EmptyResponse()

    # ==========================================================
    # Main loop
    # ==========================================================
    def spin(self):
        rospy.loginfo("[ObjectManager] In attesa di centroidi...")
        rospy.spin()


# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":
    rospy.init_node("object_manager")
    try:
        manager = ObjectManager()
        manager.spin()
    except rospy.ROSInterruptException:
        pass
