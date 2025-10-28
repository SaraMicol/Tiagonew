#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import MarkerArray, Marker
from cv_utils import  _transform_point_xyz
from cv_utils import *
##
class CentroidFilter:
    def __init__(self):
        rospy.init_node('centroid_filter', anonymous=True)
        
        # Dizionario per salvare i centroidi già visti: {label: (x, y, z)}
        self.seen_centroids = {}
        
        # Soglia di distanza (in metri)
        self.distance_threshold = 0.01
        
        # Sottoscrizione al topic dei marker in ingresso
        rospy.Subscriber('/centroid_markers', MarkerArray, self.callback)
        
        # Publisher per i marker filtrati
        self.pub = rospy.Publisher('/filtered_markers', MarkerArray, queue_size=10,latch=True)
        
        rospy.loginfo("Filtro centroidi attivo!")
    
    def distance(self, pos1, pos2):
        """Calcola la distanza euclidea tra due posizioni"""
        return ((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2 + (pos1[2]-pos2[2])**2)**0.5
    
    def callback(self, msg):
            new_markers = MarkerArray()

            for marker in msg.markers:
                #pubblico solo le spheres 
                if marker.type == Marker.SPHERE:
                    # Estrai il label
                    label = marker.ns if marker.ns else str(marker.id)
                    
                    # Posizione del centroide ORIGINALE
                    pos = (marker.pose.position.x, marker.pose.position.y, marker.pose.position.z)
                    
                    # TRASFORMA SUBITO in "map"
                    transformed_pos = _transform_point_xyz(pos,
                                                            source_frame=marker.header.frame_id,
                                                            target_frame="map")
                    
                    rospy.loginfo(f"Posizione originale per {label}: {pos}")
                    rospy.loginfo(f"Posizione trasformata per {label}: {transformed_pos}")
                    
                    # Controlla se è un centroide nuovo
                    is_new = True
                    
                    if label in self.seen_centroids:
                        dist = self.distance(transformed_pos, self.seen_centroids[label])
                        rospy.loginfo(f"dstance {dist}")
                        if dist < self.distance_threshold:
                            is_new = False
                            rospy.loginfo(f"Marker {label} già visto (distanza: {dist:.3f}m)")

                    if is_new:
                        # Salva il nuovo centroide
                        rospy.loginfo(f"Nuovo marker {label} in posizione trasformata {transformed_pos}")

                        # Crea il marker con la posizione trasformata

                        # --- Marker SFERA ---
                        sphere_marker = Marker()
                        sphere_marker.header.frame_id = "map"
                        sphere_marker.header.stamp = rospy.Time.now()
                        sphere_marker.ns = "centroid_spheres"
                        sphere_marker.id = marker.id
                        sphere_marker.type = Marker.SPHERE
                        sphere_marker.action = Marker.ADD
                        sphere_marker.pose.position.x = transformed_pos[0]
                        sphere_marker.pose.position.y = transformed_pos[1]
                        sphere_marker.pose.position.z = transformed_pos[2]
                        sphere_marker.pose.orientation.w = 1.0
                        sphere_marker.scale.x = sphere_marker.scale.y = sphere_marker.scale.z = 0.06
                        sphere_marker.color.r = 1.0
                        sphere_marker.color.g = 0.0
                        sphere_marker.color.b = 0.0
                        sphere_marker.color.a = 1.0
                        sphere_marker.lifetime=rospy.Duration(0)
                    
                        new_markers.markers.append(sphere_marker)
                        
                        # Aggiungi il centroide alla lista di quelli già visti
                        self.seen_centroids[label] = transformed_pos
                    
            # Pubblica FUORI dal loop
            if new_markers.markers:
               self.pub.publish(new_markers.markers)


if __name__ == '__main__':
    try:
        CentroidFilter()
        rospy.spin()
    except rospy.ROSInterruptedException:
        pass


