#!/usr/bin/env python3

import rospy
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import String, Empty
from std_srvs.srv import Empty as EmptyService, EmptyResponse
from geometry_msgs.msg import Point, PointStamped, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from collections import deque
from scipy.spatial.transform import Rotation

from manipulation_challenge.msg import FilteredObjectPercept as FilteredObjectPerceptMsg
from manipulation_challenge.srv import GetAllObjects, GetAllObjectsResponse
from manipulation_challenge.srv import GetObject, GetObjectRequest, GetObjectResponse
from manipulation_challenge.srv import GetCurrentlyHolding, GetCurrentlyHoldingResponse
from manipulation_challenge.srv import SetCurrentlyHolding, SetCurrentlyHoldingRequest, SetCurrentlyHoldingResponse
import tf2_ros
import tf2_py as tf2
import tf2_geometry_msgs
import sys
import os

from cv_utils import point_to_rviz

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

# === TF helpers ===
def init_tf_buffer():
    """Initialize TF2 buffer and listener for transformations"""
    global _tf_buffer, _tf_listener
    
    if _tf_buffer is None:
        _tf_buffer = tf2_ros.Buffer()
        _tf_listener = tf2_ros.TransformListener(_tf_buffer)
        rospy.loginfo("[init_tf_buffer] TF2 buffer and listener initialized")
        rospy.sleep(0.5)  # Give it time to fill buffer

def transform_point(point, source_frame, target_frame, timeout=2.0):
    """
    Transform a point (x, y, z) tuple from source_frame to target_frame.
    
    Args:
        point: Tuple of (x, y, z) coordinates
        source_frame: Source coordinate frame (e.g., "map")
        target_frame: Target coordinate frame (e.g., "odom", "base_footprint")
        timeout: Timeout for transform lookup in seconds
        
    Returns:
        Tuple of (x, y, z) in target frame, or None on failure
    """
    global _tf_buffer, _tf_listener
    
    # Initialize TF buffer if needed
    if _tf_buffer is None:
        init_tf_buffer()
    
    try:
        # Create PointStamped message
        point_stamped = PointStamped()
        point_stamped.header.frame_id = source_frame
        point_stamped.header.stamp = rospy.Time(0)  # Use latest available transform
        point_stamped.point.x = float(point[0])
        point_stamped.point.y = float(point[1])
        point_stamped.point.z = float(point[2])
        
        # Wait for transform to be available
        if not _tf_buffer.can_transform(
            target_frame, 
            source_frame, 
            rospy.Time(0), 
            rospy.Duration(timeout)
        ):
            rospy.logerr(f"[transform_point] Transform from {source_frame} to {target_frame} not available")
            rospy.logerr(f"[transform_point] Check TF tree with: rosrun tf view_frames")
            return None
        
        # Transform the point using tf2
        # Note: Requires tf2_geometry_msgs to be imported for this to work
        transformed_point = _tf_buffer.transform(point_stamped, target_frame, rospy.Duration(timeout))
        
        # Extract coordinates
        result = (
            transformed_point.point.x,
            transformed_point.point.y,
            transformed_point.point.z
        )
        
        rospy.loginfo(f"[transform_point] Transformed: {source_frame}({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) "
                     f"-> {target_frame}({result[0]:.3f}, {result[1]:.3f}, {result[2]:.3f})")
        return result
        
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
            tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"[transform_point] TF2 exception: {e}")
        rospy.logerr(f"[transform_point] This usually means the transform is not being published")
        rospy.logerr(f"[transform_point] Check with: rostopic echo /tf | grep '{source_frame}\\|{target_frame}'")
        return None
    except Exception as e:
        rospy.logerr(f"[transform_point] Unexpected error: {e}")
        rospy.logerr(f"[transform_point] Type: {type(e).name}")
        import traceback
        traceback.print_exc()
        return None

def _compute_object_centroid(object_id, timeout=2.0):
    """Compute object centroid from point cloud"""
    try:
        pc_msg = rospy.wait_for_message(f'{object_id}', PointCloud2, timeout=timeout)
        points = []
        for p in point_cloud2.read_points(pc_msg, skip_nans=True):
            points.append([p[0], p[1], p[2]])
        
        if not points:
            rospy.logerr(f"[_compute_object_centroid] No points for {object_id}")
            return None
        
        centroid = np.mean(np.array(points), axis=0)
        centroid_in_map = transform_point(
            centroid, 
            source_frame=pc_msg.header.frame_id, 
            target_frame="map", 
            timeout=timeout
        )
        
        if centroid_in_map is None:
            rospy.logerr(f"[_compute_object_centroid] Failed to transform centroid to map frame")
            return None
        
        point_to_rviz(centroid_in_map, frame_id="map", topic="/grasping_point")
        rospy.loginfo(f"[_compute_object_centroid] Centroid for {object_id}: {centroid_in_map}")
        return centroid_in_map
    except Exception as e:
        rospy.logerr(f"[_compute_object_centroid] Error: {e}")
        return None

@dataclass
class OrientedBoundingBox:
    """Represents an oriented bounding box with pose and dimensions."""
    center: np.ndarray  # 3D center position (x, y, z)
    dimensions: np.ndarray  # Box dimensions (length, width, height)
    rotation: np.ndarray  # 3x3 rotation matrix
    quaternion: Tuple[float, float, float, float]  # (x, y, z, w) orientation
    
    def to_axis_aligned(self) -> Tuple[float, float, float, float, float, float]:
        """Convert to axis-aligned bounding box format (x_min, y_min, z_min, x_max, y_max, z_max)."""
        # Get the 8 corners of the oriented bounding box
        half_dims = self.dimensions / 2.0
        corners_local = np.array([
            [-half_dims[0], -half_dims[1], -half_dims[2]],
            [half_dims[0], -half_dims[1], -half_dims[2]],
            [-half_dims[0], half_dims[1], -half_dims[2]],
            [half_dims[0], half_dims[1], -half_dims[2]],
            [-half_dims[0], -half_dims[1], half_dims[2]],
            [half_dims[0], -half_dims[1], half_dims[2]],
            [-half_dims[0], half_dims[1], half_dims[2]],
            [half_dims[0], half_dims[1], half_dims[2]],
        ])
        
        # Transform corners to world frame
        corners_world = (self.rotation @ corners_local.T).T + self.center
        
        # Compute axis-aligned bounding box
        min_coords = np.min(corners_world, axis=0)
        max_coords = np.max(corners_world, axis=0)
        
        return (float(min_coords[0]), float(min_coords[1]), float(min_coords[2]),
                float(max_coords[0]), float(max_coords[1]), float(max_coords[2]))


@dataclass
class ObjectPercept:
    """A dataclass containing object perception data."""
    type_label: str
    pcl: PointCloud2
    bounding_box: OrientedBoundingBox  # Oriented bounding box using PCA
    centroid: np.ndarray  # 3D centroid position

    def compute_bounding_box(self) -> OrientedBoundingBox:
        """Compute oriented bounding box from point cloud using PCA."""
        try:
            points = []
            for p in point_cloud2.read_points(self.pcl, skip_nans=True):
                points.append([p[0], p[1], p[2]])
            
            if not points:
                rospy.logwarn(f"No points found in point cloud for {self.type_label}")
                return OrientedBoundingBox(
                    center=np.array([0.0, 0.0, 0.0]),
                    dimensions=np.array([0.0, 0.0, 0.0]),
                    rotation=np.eye(3),
                    quaternion=(0.0, 0.0, 0.0, 1.0)
                )
            
            points_array = np.array(points)
            return _compute_pca_oriented_bounding_box(points_array)
        except Exception as e:
            rospy.logerr(f"Error computing bounding box for {self.type_label}: {e}")
            return OrientedBoundingBox(
                center=np.array([0.0, 0.0, 0.0]),
                dimensions=np.array([0.0, 0.0, 0.0]),
                rotation=np.eye(3),
                quaternion=(0.0, 0.0, 0.0, 1.0)
            )


class FilteredObjectPercept(ObjectPercept):
    """An ObjectPercept subclass with filtering capabilities.
    """

    def __init__(self, type_label: str, pcl: PointCloud2, bounding_box: OrientedBoundingBox, centroid: np.ndarray, max_history: int = 5):
        # Use the provided bounding box and centroid directly (no recompute)
        super().__init__(type_label, pcl, bounding_box, centroid)

        self.history: deque[ObjectPercept] = deque(maxlen=max_history)
        self.filtered_bounding_box: OrientedBoundingBox = bounding_box
        self.filtered_pose: np.ndarray = centroid
    
    def _compute_centroid(self, object_label: str) -> np.ndarray:
        """Compute centroid using the function from primitive_actions.py"""
        try:
            centroid = _compute_object_centroid(object_label, timeout=2.0)
            return centroid if centroid is not None else np.array([0.0, 0.0, 0.0])
        except Exception as e:
            rospy.logwarn(f"Error computing centroid for {object_label}: {e}")
            return np.array([0.0, 0.0, 0.0])
    
    def add_percept(self, percept: ObjectPercept):
        """Add a new ObjectPercept to the history."""
        self.history.append(percept)
        self._update_filtered_values()
    
    def _update_filtered_values(self):
        """Update filtered bounding box and pose based on current filtering algorithm."""
        if not self.history:
            return
        
        # Current filtering algorithm: use the latest (FIFO) instance
        latest_percept = self.history[-1]
        self.filtered_bounding_box = latest_percept.bounding_box
        self.filtered_pose = latest_percept.centroid


class ObjectManager:
    """Manages object perception with filtering capabilities."""
    
    def __init__(self, max_history: int = 10):
            self.max_history = max_history

            # object_name -> {'subscriber': sub, 'last_n_percepts': deque}
            self.objects = {}
            # filtered objects are stored here only when store_current_objects() is called
            # populated as: object_name -> FilteredObjectPercept
            self.filtered_objects = {}

            # Names from the most recent /detected_object_names message
            # Used by store_current_objects() to decide which objects to persist
            self.last_detected_objects_names = []

            self.currently_holding = None  # Track currently held object
            self.detected_names_sub = rospy.Subscriber('/detected_object_names', String, self._detected_names_callback)

            # Publisher for filtered object markers (cyan centroids)
            self.filtered_markers_pub = rospy.Publisher('/filtered_object_markers', MarkerArray, queue_size=1, latch=True)

            # Publisher for filtered object bounding boxes
            self.filtered_object_bounding_boxes_markers_pub = rospy.Publisher('/filtered_object_bounding_boxes', MarkerArray, queue_size=1, latch=True)

            # Publisher for filtered object visualization data (for drawing on image)
            self.filtered_viz_pub = rospy.Publisher('/filtered_object_viz', String, queue_size=1, latch=True)

            # Create service for store_current_objects
            self.store_objects_service = rospy.Service('/object_manager/store_current_objects', EmptyService, self._store_objects_service_handler)

            # Create services for currently_holding management
            rospy.Service('/object_manager/set_holding', SetCurrentlyHolding, self._set_holding_service_handler)
            rospy.Service('/object_manager/clear_holding', EmptyService, self._clear_holding_service_handler)
            rospy.Service('/object_manager/get_holding', GetCurrentlyHolding, self._get_holding_service_handler)

            # Create services for object retrieval
            rospy.Service('/object_manager/get_all_objects', GetAllObjects, self._get_all_objects_service_handler)
            rospy.Service('/object_manager/get_object', GetObject, self._get_object_service_handler)

            rospy.loginfo("ObjectManager initialized")
    
    def _detected_names_callback(self, msg: String):
        """Callback for detected object names topic."""
        try:
            object_names = [name.strip() for name in msg.data.split(',') if name.strip()]

            # Save the last detected list for store_current_objects() to use
            self.last_detected_objects_names = object_names

            for obj_name in object_names:
                if obj_name not in self.objects:
                    self._create_object_entry(obj_name)
                    
        except Exception as e:
            rospy.logerr(f"Error processing detected names: {e}")
        
        # Print tracked objects with latest centroid and timestamp when available
        tracked_info = []
        for name, data in self.objects.items():
            centroid = None
            ts = None
            # Prefer last_n_percepts latest entry for timestamp and centroid
            last_percepts = data.get('last_n_percepts')
            if last_percepts and len(last_percepts) > 0:
                last = last_percepts[-1]
                centroid = getattr(last, 'centroid', None)
                try:
                    ts = last.pcl.header.stamp.to_sec()
                except Exception:
                    ts = None            

            tracked_info.append((name, centroid, ts))

        print("Current objects:")
        for key, value in self.objects.items():
            print(f" - {key} ")
        print("Currently tracked objects (name, centroid, timestamp_sec):")
        for info in tracked_info:
            print(info)
        print("Currently holding:", self.currently_holding)
    
    def _create_object_entry(self, object_name: str):
        """Create a new object entry with subscriber and filtered object."""
        try:
            # Create subscriber for the object's point cloud topic
            topic_name = f'/{object_name.replace(" ","_")}'
            subscriber = rospy.Subscriber(topic_name, PointCloud2, 
                                        lambda msg, name=object_name: self._object_pcl_callback(msg, name))
            
            # Initialize the entry in objects dictionary
            self.objects[object_name] = {
                'subscriber': subscriber,
                'last_n_percepts': deque(maxlen=self.max_history)
            }
            
            rospy.loginfo(f"Created object entry for: {object_name}")
            # Try to grab a recent/latched message immediately so we don't wait for the next publish
            try:
                recent_msg = rospy.wait_for_message(topic_name, PointCloud2, timeout=0.5)
                if recent_msg is not None:
                    # call the callback to initialize the stored percepts
                    self._object_pcl_callback(recent_msg, object_name)
            except Exception:
                # no recent message available, proceed normally
                pass
            
        except Exception as e:
            rospy.logerr(f"Error creating object entry for {object_name}: {e}")
    
    def _object_pcl_callback(self, pcl_msg: PointCloud2, object_name: str):
        """Callback for individual object point cloud messages."""
        try:
            if object_name not in self.objects:
                return
            
            # Compute bounding box and centroid in map frame
            bounding_box_map = _compute_bounding_box_in_map(pcl_msg)
            centroid_map = _compute_centroid_from_pcl(pcl_msg)

            # Create ObjectPercept with map frame coordinates
            percept = ObjectPercept(object_name, pcl_msg, bounding_box_map, centroid_map)
            
            # Add to history
            self.objects[object_name]['last_n_percepts'].append(percept)
            
           
            
            # Publish updated filtered markers
            self._publish_filtered_markers()
            
            # Publish updated filtered visualization data
            self._publish_filtered_viz_data()
                
        except Exception as e:
            rospy.logerr(f"Error processing PCL for {object_name}: {e}")
    
    def store_current_objects(self):
        """Store the current objects according to the filtering algorithm.
        
        Currently: for each tracked object, store the latest percept (most recent instance)
        into the `self.filtered_objects` dictionary. This dictionary is only populated when
        this method is called.
        """
        # Reset filtered_objects and only store those in last_detected_objects_names
        # Only reset filtered_objects for non-table objects
        # Preserve the table object if it exists
        table_obj = self.filtered_objects.get('table')
        self.filtered_objects = {}
        if table_obj is not None:
            self.filtered_objects['table'] = table_obj
        try:
            names_to_store = []
            print("Last detected object names:", self.last_detected_objects_names)
            if getattr(self, 'last_detected_objects_names', None):
                names_to_store = list(self.last_detected_objects_names)
            else:
                # Fallback: store none if we have no detected list
                names_to_store = []
            print("names to store:", names_to_store)

            for obj_name in names_to_store:
                if obj_name not in self.objects:
                    rospy.logwarn(f"Attempting to store '{obj_name}' but it is not currently tracked")
                    continue

                obj_data = self.objects[obj_name]
                last_n = obj_data.get('last_n_percepts')
                if last_n and len(last_n) > 0:
                    # Use the latest percept instance
                    latest_percept = last_n[-1]

                    # Create a FilteredObjectPercept using the latest percept and store it
                    fil = FilteredObjectPercept(obj_name, latest_percept.pcl, latest_percept.bounding_box, latest_percept.centroid, max_history=self.max_history)
                    # Add the latest percept into its history so filtered fields are set
                    fil.add_percept(latest_percept)
                    self.filtered_objects[obj_name] = fil

                    point_to_rviz(latest_percept.centroid, frame_id="map", topic="/grasping_point")

                    rospy.loginfo(f"Stored current object for: {obj_name}")

        except Exception as e:
            rospy.logerr(f"Error storing current objects: {e}")

        # Print stored filtered objects
        print("Stored filtered objects:")
        for name, filtered_obj in self.filtered_objects.items():
            print(f" - {name}: centroid={filtered_obj.filtered_pose}, bounding_box={filtered_obj.filtered_bounding_box}")
    
    def _store_objects_service_handler(self, req):
        """Service handler for store_current_objects service."""
        print("Storing objects")
        try:
            self.store_current_objects()
            rospy.loginfo("store_current_objects service called successfully")
            return EmptyResponse()
        except Exception as e:
            rospy.logerr(f"Error in store_current_objects service: {e}")
            return EmptyResponse()
    
    def _set_holding_service_handler(self, req):
        """Service handler for setting currently_holding object."""
        try:
            self.currently_holding = req.object_name if req.object_name else None
            rospy.loginfo(f"Set currently holding: {self.currently_holding}")
            return SetCurrentlyHoldingResponse(success=True, message=f"Set holding to {self.currently_holding}")
        except Exception as e:
            rospy.logerr(f"Error in set_holding service: {e}")
            return SetCurrentlyHoldingResponse(success=False, message=str(e))
    
    def _clear_holding_service_handler(self, req):
        """Service handler for clearing currently_holding object."""
        try:
            self.currently_holding = None
            rospy.loginfo("Reset currently holding to None")
            return EmptyResponse()
        except Exception as e:
            rospy.logerr(f"Error in clear_holding service: {e}")
            return EmptyResponse()
    
    def _get_holding_service_handler(self, req):
        """Service handler for getting currently_holding object."""
        try:
            is_holding = self.currently_holding is not None
            object_name = self.currently_holding if self.currently_holding else ""
            rospy.loginfo(f"Currently holding: {object_name if object_name else 'None'}")
            return GetCurrentlyHoldingResponse(is_holding=is_holding, object_name=object_name)
        except Exception as e:
            rospy.logerr(f"Error in get_holding service: {e}")
            return GetCurrentlyHoldingResponse(is_holding=False, object_name="")
    
    def get_filtered_object(self, object_name: str) -> Optional[FilteredObjectPercept]:
        """Get the filtered object for a given name."""
        return self.filtered_objects.get(object_name)

    def get_object_history(self, object_name: str) -> Optional[List[ObjectPercept]]:
        """Get the perception history for a given object."""
        if object_name in self.objects:
            return list(self.objects[object_name].get('last_n_percepts', []))
        return None
    
    def get_all_objects(self) -> Dict[str, FilteredObjectPercept]:
        """Get all currently tracked filtered objects."""
        result = {}
        # Return the filtered objects stored via store_current_objects()
        for obj_name, filtered_obj in self.filtered_objects.items():
            result[obj_name] = filtered_obj
        return result

    def _filtered_object_to_msg(self, filtered_obj: FilteredObjectPercept) -> 'FilteredObjectPerceptMsg':
        """Convert FilteredObjectPercept to ROS message format."""
        try:
            msg = FilteredObjectPerceptMsg()
            msg.type_label = filtered_obj.type_label
            msg.pcl = filtered_obj.pcl
            
            # Convert OrientedBoundingBox to message format
            if hasattr(filtered_obj, 'bounding_box') and filtered_obj.bounding_box is not None:
                obb = filtered_obj.bounding_box
                
                # Set center
                msg.bounding_box_center = Point()
                msg.bounding_box_center.x = float(obb.center[0])
                msg.bounding_box_center.y = float(obb.center[1])
                msg.bounding_box_center.z = float(obb.center[2])
                
                # Set orientation from quaternion
                msg.bounding_box_orientation = Quaternion(
                    x=obb.quaternion[0],
                    y=obb.quaternion[1],
                    z=obb.quaternion[2],
                    w=obb.quaternion[3]
                )
                
                # Set dimensions
                msg.bounding_box_dimensions = [
                    float(obb.dimensions[0]),
                    float(obb.dimensions[1]),
                    float(obb.dimensions[2])
                ]
            
            # Convert filtered OrientedBoundingBox to message format
            if hasattr(filtered_obj, 'filtered_bounding_box') and filtered_obj.filtered_bounding_box is not None:
                obb = filtered_obj.filtered_bounding_box
                
                # Set center
                msg.filtered_bounding_box_center = Point()
                msg.filtered_bounding_box_center.x = float(obb.center[0])
                msg.filtered_bounding_box_center.y = float(obb.center[1])
                msg.filtered_bounding_box_center.z = float(obb.center[2])
                
                # Set orientation from quaternion
                msg.filtered_bounding_box_orientation = Quaternion(
                    x=obb.quaternion[0],
                    y=obb.quaternion[1],
                    z=obb.quaternion[2],
                    w=obb.quaternion[3]
                )
                
                # Set dimensions
                msg.filtered_bounding_box_dimensions = [
                    float(obb.dimensions[0]),
                    float(obb.dimensions[1]),
                    float(obb.dimensions[2])
                ]
            
            # Convert numpy arrays to Point messages
            msg.centroid = Point()
            msg.centroid.x = float(filtered_obj.centroid[0])
            msg.centroid.y = float(filtered_obj.centroid[1])
            msg.centroid.z = float(filtered_obj.centroid[2])
            
            msg.filtered_pose = Point()
            msg.filtered_pose.x = float(filtered_obj.filtered_pose[0])
            msg.filtered_pose.y = float(filtered_obj.filtered_pose[1])
            msg.filtered_pose.z = float(filtered_obj.filtered_pose[2])
            
            return msg
        except Exception as e:
            rospy.logerr(f"Error converting FilteredObjectPercept to message: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return FilteredObjectPerceptMsg()

    def _get_all_objects_service_handler(self, req):
        """Service handler for getting all stored objects."""
        try:
            response = GetAllObjectsResponse()
            objects = []
            object_names = []
            # Use self.filtered_objects which is populated only when store_current_objects() is called
            for obj_name, filtered_obj in self.filtered_objects.items():
                if filtered_obj is not None:
                    print(filtered_obj.filtered_pose)
                    objects.append(self._filtered_object_to_msg(filtered_obj))
                    print(objects[-1])
                    object_names.append(obj_name)
            
            response.objects = objects
            response.object_names = object_names
            rospy.loginfo(f"Returning {len(objects)} stored objects")
            return response
        except Exception as e:
            rospy.logerr(f"Error in get_all_objects service: {e}")
            return GetAllObjectsResponse()

    def _get_object_service_handler(self, req):
        """Service handler for getting a single object."""
        try:
            response = GetObjectResponse()
            
            filtered_obj = self.filtered_objects.get(req.object_name)
            if filtered_obj is not None:
                response.success = True
                response.object = self._filtered_object_to_msg(filtered_obj)
                rospy.loginfo(f"Returning object: {req.object_name}")
            else:
                response.success = False
                rospy.logwarn(f"Object {req.object_name} not found or has no filtered data")
            
            return response
        except Exception as e:
            rospy.logerr(f"Error in get_object service: {e}")
            response = GetObjectResponse()
            response.success = False
            return response
    
    def is_holding_object(self) -> bool:
        """Check if currently holding an object."""
        return self.currently_holding is not None
    
    def get_container_objects(self) -> List[str]:
        """Get list of objects that contain '_container' in their name."""
        containers = []
        for obj_name in self.objects.keys():
            if '_container' in obj_name:
                containers.append(obj_name)
        return containers
    
    def _create_bounding_box_marker(self, object_name: str, obb: OrientedBoundingBox, 
                                   marker_id: int, frame_id: str = "map") -> Marker:
        """Create an oriented bounding box marker for visualization."""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "filtered_objects_bboxes"  # Use consistent namespace
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        # Set position to OBB center
        marker.pose.position.x = float(obb.center[0])
        marker.pose.position.y = float(obb.center[1])
        marker.pose.position.z = float(obb.center[2])
        
        # Set orientation using quaternion from PCA
        marker.pose.orientation.x = obb.quaternion[0]
        marker.pose.orientation.y = obb.quaternion[1]
        marker.pose.orientation.z = obb.quaternion[2]
        marker.pose.orientation.w = obb.quaternion[3]
        
        # Set scale to OBB dimensions
        marker.scale.x = float(obb.dimensions[0])
        marker.scale.y = float(obb.dimensions[1])
        marker.scale.z = float(obb.dimensions[2])
        
        # Set cyan color for filtered objects
        marker.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.3)  # Cyan with transparency
        marker.lifetime = rospy.Duration(0)  # Permanent
        
        return marker
    
    def _publish_filtered_markers(self):
        """Publish markers for all filtered objects."""
        marker_array = MarkerArray()
        bbox_marker_array = MarkerArray()
        marker_id = 0
        
        # Iterate over filtered_objects which are populated only when store_current_objects() is called
        for object_name, filtered_obj in self.filtered_objects.items():
            if filtered_obj is not None and getattr(filtered_obj, 'history', None):
                # Use the filtered centroid (already in map coordinates)
                centroid = getattr(filtered_obj, 'filtered_pose', getattr(filtered_obj, 'centroid', None))
                bounding_box = getattr(filtered_obj, 'filtered_bounding_box', None)
                
                if centroid is None:
                    continue

                # Create a cyan sphere marker at centroid in map frame
                marker = Marker()
                marker.header.frame_id = 'map'
                marker.header.stamp = rospy.Time.now()
                marker.ns = "filtered_objects_centroids"
                marker.id = marker_id
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = float(centroid[0])
                marker.pose.position.y = float(centroid[1])
                marker.pose.position.z = float(centroid[2])
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                marker.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.9)  # cyan
                marker.lifetime = rospy.Duration(0)
                marker_array.markers.append(marker)
                
                # Create bounding box marker if available
                if bounding_box is not None:
                    bbox_marker = self._create_bounding_box_marker(
                        object_name, 
                        bounding_box, 
                        marker_id, 
                        frame_id="map"
                    )
                    bbox_marker_array.markers.append(bbox_marker)
                
                marker_id += 1
        
        # Clear old markers if we have fewer objects now
        max_markers = 50
        while marker_id < max_markers:
            # delete centroid markers
            clear_marker_c = Marker()
            clear_marker_c.header.frame_id = "map"
            clear_marker_c.header.stamp = rospy.Time.now()
            clear_marker_c.ns = "filtered_objects_centroids"
            clear_marker_c.id = marker_id
            clear_marker_c.action = Marker.DELETE
            marker_array.markers.append(clear_marker_c)

            # delete bounding box markers with correct namespace
            clear_marker_b = Marker()
            clear_marker_b.header.frame_id = "map"
            clear_marker_b.header.stamp = rospy.Time.now()
            clear_marker_b.ns = "filtered_objects_bboxes"
            clear_marker_b.id = marker_id
            clear_marker_b.action = Marker.DELETE
            bbox_marker_array.markers.append(clear_marker_b)

            marker_id += 1
        
        # Publish both marker arrays
        self.filtered_markers_pub.publish(marker_array)
        self.filtered_object_bounding_boxes_markers_pub.publish(bbox_marker_array)
    
    def _publish_filtered_viz_data(self):
        """Publish filtered object names for visualization overlay."""
        filtered_names = []
        # Use only objects stored in self.filtered_objects (populated via store_current_objects())
        for object_name, filtered_obj in self.filtered_objects.items():
            if filtered_obj is not None and getattr(filtered_obj, 'history', None):
                filtered_names.append(object_name)
        
        if filtered_names:
            viz_data = ','.join(filtered_names)
            self.filtered_viz_pub.publish(String(data=viz_data))

def _compute_pca_oriented_bounding_box(points: np.ndarray, rotate_only_z: bool = False) -> OrientedBoundingBox:
    """
    Compute an oriented bounding box using PCA.
    
    Args:
        points: Nx3 numpy array of 3D points
        rotate_only_z: If True, only rotate around Z axis (keep XY axes aligned with world frame)
        
    Returns:
        OrientedBoundingBox containing the OBB parameters
    """
    if len(points) < 3:
        rospy.logwarn("Need at least 3 points for PCA, returning default OBB")
        return OrientedBoundingBox(
            center=np.array([0.0, 0.0, 0.0]),
            dimensions=np.array([0.0, 0.0, 0.0]),
            rotation=np.eye(3),
            quaternion=(0.0, 0.0, 0.0, 1.0)
        )
    
    # Compute centroid
    centroid = np.mean(points, axis=0)
    
    # Center the points
    centered_points = points - centroid
    
    if rotate_only_z:
        # Only consider XY plane for rotation
        centered_points_xy = centered_points[:, :2]
        cov_matrix_xy = np.cov(centered_points_xy.T)
        
        # Compute eigenvalues and eigenvectors for 2D
        eigenvalues_xy, eigenvectors_xy = np.linalg.eig(cov_matrix_xy)
        
        # Sort by eigenvalues
        sorted_indices = np.argsort(eigenvalues_xy)[::-1]
        eigenvectors_xy = eigenvectors_xy[:, sorted_indices]
        
        # Build 3D rotation matrix (rotation only around Z)
        rotation_matrix = np.eye(3)
        rotation_matrix[:2, :2] = eigenvectors_xy
        
        # Ensure right-handed coordinate system
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix[:2, 1] *= -1
    else:
        # Full 3D PCA
        # Compute covariance matrix
        cov_matrix = np.cov(centered_points.T)
        
        # Perform eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort eigenvectors by eigenvalues (descending order)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        eigenvalues = eigenvalues[sorted_indices]
        
        # Ensure right-handed coordinate system
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] *= -1
        
        # Rotation matrix (columns are principal axes)
        rotation_matrix = eigenvectors
    
    # Project points onto principal axes
    projected_points = centered_points @ rotation_matrix
    
    # Compute dimensions along principal axes
    min_proj = np.min(projected_points, axis=0)
    max_proj = np.max(projected_points, axis=0)
    dimensions = max_proj - min_proj
    
    # Adjust center to account for non-symmetric projection
    center_offset = (max_proj + min_proj) / 2.0
    adjusted_center = centroid + rotation_matrix @ center_offset
    
    # Convert rotation matrix to quaternion using scipy
    rotation = Rotation.from_matrix(rotation_matrix)
    quat = rotation.as_quat()  # Returns [x, y, z, w]
    
    return OrientedBoundingBox(
        center=adjusted_center,
        dimensions=dimensions,
        rotation=rotation_matrix,
        quaternion=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    )



def _compute_centroid_from_pcl(pcl_msg: PointCloud2, timeout=5.0, tf_buffer=None) -> np.ndarray:
    points = []
    #print(pcl_msg)
    for p in point_cloud2.read_points(pcl_msg, skip_nans=True):
        points.append([p[0], p[1], p[2], p[3]])  # x,y,z,label
        #print("appending point:", points[-1])
    #print(points)
    centroid = np.mean(np.array(points), axis=0)[:3]  # (x,y,z)
    #print("centroid:", centroid)
    if tf_buffer is None:
        tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(tf_buffer)
    #print("frame of centroid:", pcl_msg.header.frame_id)
    # Wait for transform to be available, then use Time(0) to get the latest
    # This avoids extrapolation errors from timing issues
    if tf_buffer.can_transform("map", pcl_msg.header.frame_id, rospy.Time(0), rospy.Duration(timeout)):
        trans = tf_buffer.lookup_transform("map", pcl_msg.header.frame_id, rospy.Time(0))
    else:
        raise Exception(f"Transform from {pcl_msg.header.frame_id} to map not available after {timeout}s")
    # Transform centroid to map frame
    # Create a PointStamped in the source frame so tf2 can transform it
    centroid_stamped = PointStamped()
    centroid_stamped.header.frame_id = pcl_msg.header.frame_id
    centroid_stamped.header.stamp = rospy.Time(0)  # Use latest available transform
    centroid_stamped.point.x = float(centroid[0])
    centroid_stamped.point.y = float(centroid[1])
    centroid_stamped.point.z = float(centroid[2])

    # Apply the transformation
    transformed_point_stamped = tf2_geometry_msgs.do_transform_point(centroid_stamped, trans)

    # Extract transformed coordinates
    centroid_in_map = np.array([
        transformed_point_stamped.point.x,
        transformed_point_stamped.point.y,
        transformed_point_stamped.point.z,
    ])
    
    #R, T = _get_R_and_T(trans)
    #p = np.array(centroid).reshape(3)
    #centroid_in_map = R.T.dot(p - T)
    #print("centroid in map:", centroid_in_map)
    #point_to_rviz(centroid_in_map, frame_id="map", topic="/grasping_point")
    return centroid_in_map

def _compute_bounding_box_in_map(pcl_msg: PointCloud2, timeout=5.0, tf_buffer=None) -> OrientedBoundingBox:
    """Compute oriented bounding box using PCA in map frame coordinates."""
    try:
        points = []
        for p in point_cloud2.read_points(pcl_msg, skip_nans=True):
            points.append([p[0], p[1], p[2]])  # x,y,z only
        
        if not points:
            rospy.logwarn(f"No points found in point cloud for bounding box computation")
            return OrientedBoundingBox(
                center=np.array([0.0, 0.0, 0.0]),
                dimensions=np.array([0.0, 0.0, 0.0]),
                rotation=np.eye(3),
                quaternion=(0.0, 0.0, 0.0, 1.0)
            )
        
        points_array = np.array(points)
        
        # Get transform to map frame (same as _compute_centroid_from_pcl)
        if tf_buffer is None:
            tf_buffer = tf2_ros.Buffer()
            tf2_ros.TransformListener(tf_buffer)
        
        # Wait for transform to be available, then use Time(0) to get the latest
        # This avoids extrapolation errors from timing issues
        if tf_buffer.can_transform("map", pcl_msg.header.frame_id, rospy.Time(0), rospy.Duration(timeout)):
            trans = tf_buffer.lookup_transform("map", pcl_msg.header.frame_id, rospy.Time(0))
        else:
            raise Exception(f"Transform from {pcl_msg.header.frame_id} to map not available after {timeout}s")
        
        # Transform each point to map frame using tf2_geometry_msgs (same as centroid)
        points_in_map = []
        for point in points_array:
            # Create a PointStamped for each point
            point_stamped = PointStamped()
            point_stamped.header.frame_id = pcl_msg.header.frame_id
            point_stamped.header.stamp = rospy.Time(0)  # Use latest available transform
            point_stamped.point.x = float(point[0])
            point_stamped.point.y = float(point[1])
            point_stamped.point.z = float(point[2])
            
            # Apply the transformation using tf2_geometry_msgs
            transformed_point_stamped = tf2_geometry_msgs.do_transform_point(point_stamped, trans)
            
            # Extract transformed coordinates
            points_in_map.append([
                transformed_point_stamped.point.x,
                transformed_point_stamped.point.y,
                transformed_point_stamped.point.z
            ])
        
        points_in_map = np.array(points_in_map)
        
        # Compute oriented bounding box using PCA in map frame
        obb = _compute_pca_oriented_bounding_box(points_in_map)
        
        return obb
    except Exception as e:
        rospy.logerr(f"Error computing OBB in map frame: {e}")
        import traceback
        rospy.logerr(traceback.format_exc())
        return OrientedBoundingBox(
            center=np.array([0.0, 0.0, 0.0]),
            dimensions=np.array([0.0, 0.0, 0.0]),
            rotation=np.eye(3),
            quaternion=(0.0, 0.0, 0.0, 1.0)
        )

if __name__ == "__main__":
    rospy.init_node("object_manager_node")
    manager = ObjectManager()
    rospy.spin()