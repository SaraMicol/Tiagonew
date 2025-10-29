#!/usr/bin/env python3
import os
import rospy
import numpy as np
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from matplotlib.colors import to_rgb
import cv2
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Header, String, ColorRGBA
from sensor_msgs.msg import PointField
import json
import struct
import tf2_ros

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from tiago_project.msg import Centroid, CentroidArray


def _get_R_and_T(trans):
    Tx_base = trans.transform.translation.x
    Ty_base = trans.transform.translation.y
    Tz_base = trans.transform.translation.z
    T = np.array([Tx_base, Ty_base, Tz_base])
    # Quaternion coordinates
    qx = trans.transform.rotation.x
    qy = trans.transform.rotation.y
    qz = trans.transform.rotation.z
    qw = trans.transform.rotation.w
    
    # Rotation matrix
    R = 2*np.array([[pow(qw,2) + pow(qx,2) - 0.5, qx*qy-qw*qz, qw*qy+qx*qz],[qw*qz+qx*qy, pow(qw,2) + pow(qy,2) - 0.5, qy*qz-qw*qx],[qx*qz-qw*qy, qw*qx+qy*qz, pow(qw,2) + pow(qz,2) - 0.5]])
    return R, T

def _transform_point_xyz(pt_xyz, source_frame, target_frame, timeout=1.0):
    """Transform a 3D point (numpy array/list len=3) from source_frame to target_frame using TF."""
    if target_frame == source_frame:
        return np.array(pt_xyz).reshape(3)
    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer)
    trans = tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(timeout))
    R, T = _get_R_and_T(trans)
    p = np.array(pt_xyz).reshape(3)
    return R.dot(p) + T

def _clear_markers(topic):
    """Publish a MarkerArray with DELETEALL to clear previous markers on 'topic'."""
    ma = MarkerArray()
    m = Marker()
    m.action = Marker.DELETEALL
    ma.markers.append(m)
    pub = rospy.Publisher(topic, MarkerArray, queue_size=1, latch=True)
    # small sleep to let publisher register
    rospy.sleep(0.05)
    pub.publish(ma)


def overlay_mask_on_image(image, mask, color_rgb=(0.0, 1.0, 0.0), alpha=0.5):
    """Apply a semi-transparent mask overlay on a BGR image.

    - image: numpy array BGR uint8 (H,W,3)
    - mask: numpy array 2D uint8 (0 or 255) or bool
    - color_rgb: tuple (r,g,b) in 0..1
    - alpha: blending factor
    """
    mask_bool = (mask > 0)
    color_bgr = np.array([int(c * 255) for c in color_rgb[::-1]], dtype=np.uint8)
    img_pixels = image[mask_bool]
    blended = (img_pixels.astype(float) * (1.0 - alpha) + color_bgr.astype(float) * alpha).astype(np.uint8)
    image[mask_bool] = blended
    return image

def rectangles_overlap(rect1, rect2):
    """Check if two rectangles overlap."""
    x1_min, y1_min, x1_max, y1_max = rect1
    x2_min, y2_min, x2_max, y2_max = rect2
    
    return not (x1_max < x2_min or x2_max < x1_min or 
                y1_max < y2_min or y2_max < y1_min)


    # Soluzione 1: Posizionamento intelligente del testo
def draw_detections(img, detections):
    occupied_regions = []  # Lista di regioni già occupate
    
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.bbox)
        
        # Disegna il bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepara il testo
        text = f"{detection.label}: {detection.score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        # Prova diverse posizioni per il testo (in ordine di preferenza)
        positions = [
            (x1, y1 - text_height - 5),           # Sopra il box (default)
            (x1, y2 + text_height + 5),           # Sotto il box
            (x2 + 5, y1),                          # Destra del box
            (x1 - text_width - 5, y1),            # Sinistra del box
            (x1, y1 + text_height + 5),           # Dentro il box in alto
            ((x1 + x2) // 2 - text_width // 2, y1 - text_height - 5)  # Centrato sopra
        ]
        
        # Trova la prima posizione che non si sovrappone
        final_pos = positions[0]  # Default
        for pos_x, pos_y in positions:
            # Definisci il rettangolo del testo
            text_rect = (
                pos_x, 
                pos_y - text_height - baseline - 5,
                pos_x + text_width,
                pos_y
            )
            
            # Controlla se si sovrappone con altre label
            overlaps = False
            for occupied in occupied_regions:
                if rectangles_overlap(text_rect, occupied):
                    overlaps = True
                    break
            
            # Controlla se è dentro l'immagine
            if (text_rect[0] >= 0 and text_rect[1] >= 0 and 
                text_rect[2] < img.shape[1] and text_rect[3] < img.shape[0] and 
                not overlaps):
                final_pos = (pos_x, pos_y)
                occupied_regions.append(text_rect)
                break
        
        # Disegna sfondo e testo
        text_x, text_y = final_pos
        cv2.rectangle(
            img, 
            (text_x, text_y - text_height - baseline - 5), 
            (text_x + text_width, text_y), 
            (0, 255, 0), 
            -1
        )
        cv2.putText(
            img, 
            text, 
            (text_x, text_y - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 0), 
            2
        )
    
    path_file = os.path.dirname(os.path.abspath(__file__))
    cv2.imwrite(os.path.join(path_file, "../assets/debug_detections.jpg"), img)
    return img

def draw_filtered_detections(img, filtered_detections):
        """Draw cyan bounding boxes for filtered objects that match current detections."""
        for detection in filtered_detections:
            # Check if this detection matches a filtered object
            x1, y1, x2, y2 = map(int, detection.bbox)
            
            # Draw cyan bounding box for filtered objects (overwrite the green one)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3)  # Cyan (BGR format)
            
            # Draw cyan text background and label
            text = f"{detection.label} (stored)"
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Position text above the box
            text_x = x1
            text_y = y1 - text_height - 10
            if text_y < text_height:  # If too close to top, put it below the box
                text_y = y2 + text_height + 10
            
            # Draw cyan background for text
            cv2.rectangle(
                img, 
                (text_x, text_y - text_height - baseline - 5), 
                (text_x + text_width, text_y), 
                (255, 255, 0),  # Cyan background
                -1
            )
            # Draw black text on cyan background
            cv2.putText(
                img, text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
            )
        
        path_file = os.path.dirname(os.path.abspath(__file__))
        cv2.imwrite(os.path.join(path_file, "../assets/debug_detections.jpg"), img)
        

def point_to_rviz(point_xyz, frame_id="map", topic="/marker_point", marker_scale=0.04):
    """Publish a single sphere Marker at point_xyz (3,) in given frame_id.

    Marker is published with latch=True and permanent lifetime.
    """
    if len(point_xyz) != 3:
        raise ValueError("point_xyz must be a 3-element iterable")

    # Clear any previous markers on this topic so stale objects are removed
    try:
        _clear_markers(topic)
    except Exception:
        # If clearing fails, continue and publish the new marker anyway
        rospy.logdebug(f"_clear_markers failed for topic {topic}")

    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = rospy.Time.now()
    m.ns = "point"
    m.id = 0
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.pose.position.x = float(point_xyz[0])
    m.pose.position.y = float(point_xyz[1])
    m.pose.position.z = float(point_xyz[2])
    m.pose.orientation.w = 1.0
    m.scale.x = marker_scale
    m.scale.y = marker_scale
    m.scale.z = marker_scale
    m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
    m.lifetime = rospy.Duration(0)

    marker_array = MarkerArray()
    marker_array.markers.append(m)

    pub = rospy.Publisher(topic, MarkerArray, queue_size=1, latch=True)
    rospy.sleep(0.05)
    pub.publish(marker_array)

def points_list_to_rviz(points_2d, depth_image, camera_info, color = None, labels=None, frame_id="camera_link", topic="/markers_objects", marker_scale=0.04):
    """Publish a MarkerArray of spheres obtained from 2D points + depth.

    Markers are published with latch=True and permanent lifetime.
    """
    if labels is None:
        labels = [f"point_{i}" for i in range(len(points_2d))]

    if not isinstance(camera_info, CameraInfo):
        raise TypeError('camera_info must be sensor_msgs.msg.CameraInfo')

    K = camera_info.K
    fx = K[0]
    fy = K[4]
    cx = K[2]
    cy = K[5]

    # Clear previous markers for this topic so old detections are removed
    try:
        _clear_markers(topic)
    except Exception:
        rospy.logdebug(f"_clear_markers failed for topic {topic}")

    marker_array = MarkerArray()
    id_to_label = {}
    palette = [to_rgb(color)] if color is not None else [to_rgb("blue")]

    for marker_id, (x, y) in enumerate(points_2d):
        x_int = int(x)
        y_int = int(y)
        if x_int < 0 or y_int < 0 or y_int >= depth_image.shape[0] or x_int >= depth_image.shape[1]:
            continue
        z = float(depth_image[y_int, x_int])
        if z <= 0 or np.isnan(z):
            continue
        X = (x - cx) * z / fx
        Y = (y - cy) * z / fy
        col = palette[marker_id % len(palette)]
        color = ColorRGBA(r=float(col[0]), g=float(col[1]), b=float(col[2]), a=1.0)
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = "points"
        m.id = marker_id
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(X)
        m.pose.position.y = float(Y)
        m.pose.position.z = float(z)
        m.pose.orientation.w = 1.0
        m.scale.x = marker_scale
        m.scale.y = marker_scale
        m.scale.z = marker_scale
        m.color = color
        m.lifetime = rospy.Duration(0)
        marker_array.markers.append(m)
        id_to_label[marker_id] = labels[marker_id] if marker_id < len(labels) else f"point_{marker_id}"

    if len(marker_array.markers) == 0:
        rospy.logdebug("points_list_to_rviz: no valid markers")
        return

    pub = rospy.Publisher(topic, MarkerArray, queue_size=1, latch=True)
    rospy.sleep(0.05)
    pub.publish(marker_array)

    try:
        labels_pub = rospy.Publisher(topic + "_labels", String, queue_size=1, latch=True)
        labels_msg = String()
        labels_msg.data = json.dumps(id_to_label)
        rospy.sleep(0.01)
        labels_pub.publish(labels_msg)
    except Exception:
        rospy.logwarn("Unable to publish marker labels mapping")


def mask_list_to_pointcloud2(masks, depth_image, camera_info, labels=None, frame_id="camera_link", topic="/pcl_objects", max_points_per_obj=20000):
    """Aggregate masks into a PointCloud2 with an object_id field and publish it (latch=True).

    This function keeps object IDs deterministic per call (uses index-based IDs).
    """
    if labels is None:
        labels = [f"obj_{i}" for i in range(len(masks))]

    if not isinstance(camera_info, CameraInfo):
        raise TypeError('camera_info must be sensor_msgs.msg.CameraInfo')

    K = camera_info.K
    fx = K[0]
    fy = K[4]
    cx = K[2]
    cy = K[5]

    palette_names = ['red', 'green', 'blue', 'magenta', 'cyan', 'yellow', 'orange', 'purple', 'brown', 'pink']
    palette = [to_rgb(c) for c in palette_names]

    current_points = []
    id_to_label = {}

    for obj_idx, mask in enumerate(masks):
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        if mask.ndim == 3:
            mask2d = mask[:, :, 0]
        else:
            mask2d = mask
        mask_bool = mask2d.astype(bool)
        ys, xs = np.nonzero(mask_bool)
        if len(xs) == 0:
            continue
        if len(xs) > max_points_per_obj:
            idx = np.linspace(0, len(xs) - 1, max_points_per_obj).astype(int)
            xs = xs[idx]
            ys = ys[idx]

        col = palette[obj_idx % len(palette)]
        r = int(col[0] * 255) & 0xFF
        g = int(col[1] * 255) & 0xFF
        b = int(col[2] * 255) & 0xFF
        rgb_uint = (r << 16) | (g << 8) | b
        rgb_packed = struct.unpack('f', struct.pack('I', rgb_uint))[0]

        pts = []
        for x, y in zip(xs, ys):
            z = float(depth_image[y, x])
            if z <= 0 or np.isnan(z):
                continue
            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            pts.append((float(X), float(Y), float(z), float(rgb_packed), int(obj_idx)))

        if len(pts) == 0:
            continue

        current_points.extend(pts)
        id_to_label[int(obj_idx)] = labels[obj_idx] if obj_idx < len(labels) else f"obj_{obj_idx}"

    if len(current_points) == 0:
        rospy.logdebug("mask_list_to_pointcloud2: no points to publish")
        return

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.FLOAT32, 1),
        PointField('object_id', 16, PointField.INT32, 1)
    ]

    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    cloud_msg = point_cloud2.create_cloud(header, fields, current_points)

    pub = rospy.Publisher(topic, PointCloud2, queue_size=1, latch=True)
    rospy.sleep(0.05)
    pub.publish(cloud_msg)

    try:
        labels_pub = rospy.Publisher(topic + "_labels", String, queue_size=1, latch=True)
        labels_msg = String()
        labels_msg.data = json.dumps(id_to_label)
        rospy.sleep(0.01)
        labels_pub.publish(labels_msg)
    except Exception:
        rospy.logwarn("Unable to publish pcl labels mapping")


def publish_individual_pointclouds_by_id(masks, depth_image, camera_info, labels=None, frame_id="camera_link", topic_prefix="/pcl_id", max_points_per_obj=20000):
    """Publish one PointCloud2 per mask on topics `<topic_prefix>_<id>` (latch=True).

    Returns number of PointCloud2 published.
    """
    if labels is None:
        labels = [f"obj_{i}" for i in range(len(masks))]

    if not isinstance(camera_info, CameraInfo):
        raise TypeError('camera_info must be sensor_msgs.msg.CameraInfo')

    K = camera_info.K
    fx = K[0]
    fy = K[4]
    cx = K[2]
    cy = K[5]

    palette_names = ['red', 'green', 'blue', 'magenta', 'cyan', 'yellow', 'orange', 'purple', 'brown', 'pink']
    palette = [to_rgb(c) for c in palette_names]

    published_count = 0
    for obj_id, mask in enumerate(masks):
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        if mask.ndim == 3:
            mask2d = mask[:, :, 0]
        else:
            mask2d = mask
        mask_bool = mask2d.astype(bool)
        ys, xs = np.nonzero(mask_bool)
        if len(xs) == 0:
            continue
        if len(xs) > max_points_per_obj:
            idx = np.linspace(0, len(xs) - 1, max_points_per_obj).astype(int)
            xs = xs[idx]
            ys = ys[idx]

        col = palette[obj_id % len(palette)]
        r = int(col[0] * 255) & 0xFF
        g = int(col[1] * 255) & 0xFF
        b = int(col[2] * 255) & 0xFF
        rgb_uint = (r << 16) | (g << 8) | b
        rgb_packed = struct.unpack('f', struct.pack('I', rgb_uint))[0]

        points = []
        for x, y in zip(xs, ys):
            z = float(depth_image[y, x])
            if z <= 0 or np.isnan(z):
                continue
            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            points.append((float(X), float(Y), float(z), float(rgb_packed), int(obj_id)))

        if len(points) == 0:
            continue

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.FLOAT32, 1),
            PointField('object_id', 16, PointField.INT32, 1)
        ]

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id

        cloud_msg = point_cloud2.create_cloud(header, fields, points)
        topic_name = f"{labels[obj_id]}"
        pub = rospy.Publisher(topic_name, PointCloud2, queue_size=1, latch=True)
        rospy.sleep(0.01)
        pub.publish(cloud_msg)
        published_count += 1

    rospy.loginfo(f"publish_individual_pointclouds_by_id: published {published_count} PointCloud2")
    return published_count

def mask_list_to_centroid(masks, depth_image, camera_info, max_points_per_obj=20000):
    """Calcola il centroide 3D di ogni maschera data la profondità e la camera_info."""
    if not isinstance(camera_info, CameraInfo):
        raise TypeError('camera_info must be sensor_msgs.msg.CameraInfo')

    K = camera_info.K
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    centroids_3d = []

    for mask in masks:
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        if mask.ndim == 3:
            mask2d = mask[:, :, 0]
        else:
            mask2d = mask
        mask_bool = mask2d.astype(bool)
        ys, xs = np.nonzero(mask_bool)
        if len(xs) == 0:
            centroids_3d.append(None)
            continue

        if len(xs) > max_points_per_obj:
            idx = np.linspace(0, len(xs) - 1, max_points_per_obj).astype(int)
            xs, ys = xs[idx], ys[idx]

        # Calcolo coordinate 3D dei punti nel mask
        points_3d = []
        for x, y in zip(xs, ys):
            z = float(depth_image[y, x])
            if z <= 0 or np.isnan(z):
                continue
            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            points_3d.append((X, Y, z))

        if len(points_3d) == 0:
            centroids_3d.append(None)
            continue

        points_3d = np.array(points_3d)
        centroid = np.mean(points_3d, axis=0)
        centroids_3d.append(tuple(centroid))

    return centroids_3d


def publish_centroids_3d(centroids_3d, labels, camera_frame, target_frame="map", frame_id="map", topic="/centroids_custom"):
    """
    Pubblica i centroidi 3D già calcolati come custom message CentroidArray
    trasformati nel frame target_frame.
    
    centroids_3d: lista di tuple (X, Y, Z) o None se la maschera era vuota
    labels: lista di label corrispondenti alle maschere/detections
    camera_frame: frame di origine (dove sono calcolati i centroidi)
    target_frame: frame di destinazione per la pubblicazione
    """
    if len(centroids_3d) != len(labels):
        rospy.logwarn("Numero di centroidi e labels non corrisponde")
        return
    
     #salvataggio detections in un text file
    file_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "detections.txt")

    msg_array = CentroidArray()
    msg_array.header.stamp = rospy.Time.now()
    msg_array.header.frame_id = "map"
    points1=[]

    with open(file_path, 'a') as f:
            f.write(f"-"*60)
            f.write(f"new_cycle \n")

    for i, centroid in enumerate(centroids_3d):
        if centroid is None:
            continue # salta maschere vuote

        # Trasforma il centroide dal frame camera al frame target
        try:
            X, Y, Z = _transform_point_xyz(centroid, source_frame=camera_frame, target_frame=target_frame)
        except Exception as e:
            rospy.logwarn(f"Trasformazione centroide fallita: {e}")
            continue
        
        with open(file_path, 'a') as f:
            f.write(f"detection: {labels[i]},{X:.6f}, {Y:.6f},{Z:.6f}\n")

        centroid_msg = Centroid()
        centroid_msg.label = labels[i]
        centroid_msg.x = X
        centroid_msg.y = Y
        centroid_msg.z = Z
        points1.append(tuple((X,Y,Z)))
        msg_array.centroids.append(centroid_msg)

    pub = rospy.Publisher(topic, CentroidArray, queue_size=1, latch=True)
    rospy.sleep(0.05)
    pub.publish(msg_array)
    return points1
	

def points_list_to_rviz_3d(points, labels=None, frame_id="map", topic="/centroid_markers", marker_scale=0.06):
    """
    Visualizza una lista di punti 3D in RViz come sfere.
    
    :param points: lista di tuple (x, y, z)
    :param labels: lista di label corrispondenti (opzionale)
    :param frame_id: frame di riferimento per i marker
    :param topic: topic ROS su cui pubblicare i marker
    :param marker_scale: dimensione delle sfere in RViz
    """
    pub = rospy.Publisher(topic, MarkerArray, queue_size=1, latch=True)
    marker_array = MarkerArray()
    color = ColorRGBA(0.0, 1.0, 0.0, 1.0)  # verde 
    color_text=ColorRGBA(1.0, 1.0, 1.0, 1.0)  #bianco

    for i, point in enumerate(points):
        if point is None:
            continue

        x, y, z = point
        label= labels[i] if labels and i < len(labels) else f"obj_{i}"

        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "centroid_spheres"
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.scale.x = marker.scale.y = marker.scale.z = marker_scale
        marker.color = color
        marker.lifetime = rospy.Duration(0)

        text_marker=Marker()
        text_marker.header.frame_id=frame_id
        text_marker.header.stamp = rospy.Time.now()
        text_marker.ns = "centroid_labels"
        text_marker.id = i +1000
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = x
        text_marker.pose.position.y = y
        text_marker.pose.position.z = z + marker_scale * 1.5
        text_marker.scale.z = marker_scale * 1.2
        text_marker.color = color_text
        text_marker.text=label
        text_marker.lifetime = rospy.Duration(0)

        marker_array.markers.append(marker)
        marker_array.markers.append(text_marker)

    rospy.sleep(0.05)
    pub.publish(marker_array)
