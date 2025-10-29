#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA


def filter_marker(self, marker, label):
    #pubblico solo le spheres
    markers=[] 
     
    if marker.type == Marker.SPHERE:
        
        # Posizione del centroide ORIGINALE
        pos = (marker.pose.position.x, marker.pose.position.y, marker.pose.position.z)
        rospy.loginfo(f"help {marker.header.frame_id}")
        
        rospy.loginfo(f"Posizione originale per {label}: {pos}")
        
        # Controlla se è un centroide nuovo
        is_new = True
        
        if label in self.seen_centroids:
            dist = self.distance(pos, self.seen_centroids[label])
            rospy.loginfo(f"dostance {dist}")
            if dist < self.distance_threshold:
                is_new = False
                rospy.loginfo(f"Marker {label} già visto (distanza: {dist:.3f}m)")

        if is_new:
            # Salva il nuovo centroide
            rospy.loginfo(f"Nuovo marker {label} in posizione trasformata {pos}")

            # Crea il marker con la posizione trasformata

            # --- Marker SFERA ---
            sphere_marker = Marker()
            sphere_marker.header.frame_id = "map"
            sphere_marker.header.stamp = rospy.Time.now()
            sphere_marker.ns = "centroid_spheres" #_{string(element_pub)}
            sphere_marker.id = self.element_pub + 1 
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD
            sphere_marker.pose.position.x = pos[0]
            sphere_marker.pose.position.y = pos[1]
            sphere_marker.pose.position.z = pos[2]
            sphere_marker.pose.orientation.w = 1.0
            sphere_marker.scale.x = sphere_marker.scale.y = sphere_marker.scale.z = 0.06
            sphere_marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
            sphere_marker.lifetime=rospy.Duration(0)

            text_marker=Marker()
            text_marker.header.frame_id="map"
            text_marker.header.stamp = rospy.Time.now()
            text_marker.ns = "centroid_labels_filtered"
            text_marker.id =  self.element_pub +1000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = pos[0]
            text_marker.pose.position.y = pos[1]
            text_marker.pose.position.z = pos[2] + 0.1
            text_marker.scale.z = 0.08
            text_marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
            text_marker.text=label
            text_marker.lifetime = rospy.Duration(0)
            
            self.element_pub+=1
            rospy.loginfo(f"help1:{self.element_pub}")
            #self.new_markers.markers.append(sphere_marker)
            #self.new_markers.markers.append(text_marker)
                
            # Aggiungi il centroide alla lista di quelli già visti
            self.seen_centroids[label] = pos

            markers=[sphere_marker,text_marker]
            
        #print("lenght",len(self.new_markers.markers))
        #self.pub.publish(self.new_markers)
        return markers
