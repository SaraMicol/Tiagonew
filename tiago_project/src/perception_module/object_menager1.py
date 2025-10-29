#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import MarkerArray, Marker
from cv_utils import  _transform_point_xyz
from cv_utils import *
import utils
from filter_utils import filter_marker
from tiago_project.msg import Centroid, CentroidArray
from std_msgs.msg import ColorRGBA


class CentroidFilter:
    def __init__(self):
        rospy.init_node('centroid_filter', anonymous=True)
        
        # Dizionario per salvare i centroidi già visti: {label: (x, y, z)}
        self.seen_centroids = {}
        self.stored_markers=[]
        self.new_markers=[]
        # Soglia di distanza (in metri)
        self.distance_threshold = 0.1
        self.new_markers = MarkerArray()
        self.element_pub = 0
        # Sottoscrizione al topic dei marker in ingresso
        rospy.Subscriber('/centroid_markers', MarkerArray, self.callback)
        
        # Publisher per i marker filtrati
        self.pub = rospy.Publisher('/filtered_markers', MarkerArray, queue_size=10,latch=True)
        
        rospy.loginfo("Filtro centroidi attivo!")
    
    def distance(self, pos1, pos2):
        """Calcola la distanza euclidea tra due posizioni"""
        return ((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2 + (pos1[2]-pos2[2])**2)**0.5
    
    def callback(self, msg):
            print("-"*60)
            print("new cycle")

            labels_by_id={}
            
            # Ciclo sui marker per associare le etichette correttamente
            # Ciclo sui marker per associare le etichette correttamente
            for marker in msg.markers:
                if marker.ns == "centroid_labels" and marker.type == Marker.TEXT_VIEW_FACING:
                    label_id = marker.id
                    labels_by_id[label_id] = marker.text  # Aggiungi l'etichetta al dizionario
                    print("marker_text:", marker.text)

            # Ciclo sui marker delle sfere per associare l'etichetta e aggiungere nuovi marker
            for marker in msg.markers:
                if marker.ns == "centroid_spheres" and marker.type == Marker.SPHERE:
                    marker_id = marker.id + 1000  # ID delle etichette (i + 1000)
                    print("marker_id:", marker_id)

                    # Verifica se l'ID dell'etichetta è presente nel dizionario
                    if marker_id in labels_by_id:
                        label = labels_by_id[marker_id]
                        print("label:", label)
                    else:
                        label = None  # Se non c'è una corrispondenza, imposta a None
                        print(f"Nessuna etichetta trovata per marker con ID {marker_id}")

                    # Ora puoi usare `label` in `filter_marker`
                    markers = filter_marker(self, marker, label)
                    #rospy.loginfo(f"Markers restituiti: {type(markers)}")

                    #prendo solo il centroide
                    # Verifica che markers non sia vuota prima di accedere
                    if markers:
                        # prendo solo il centroide (se markers non è vuota)
                        new_marker = markers[0]
                        self.new_markers.markers.append(new_marker)
                    else:
                        #ho bisogno di questo errore perchè viene creato un nuovo centroide solo se è nuovo 
                        rospy.logwarn(f"Nessun marker restituito per il centroide {label} perchè lo avevo già visto")
            
            rospy.loginfo(f"totale markers pubblicati: {len(self.new_markers.markers)}")      
            self.pub.publish(self.new_markers)

if __name__ == '__main__':
    try:
        CentroidFilter()
        rospy.spin()
    except rospy.ROSInterruptedException:
        pass

