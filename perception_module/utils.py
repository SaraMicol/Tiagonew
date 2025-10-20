import rospy
import numpy as np
from cv_bridge import CvBridge
import cv2
import open3d as o3d
from sensor_msgs.msg import Image, CameraInfo

bridge = CvBridge()

ENCODER_VITSAM_PATH = "/tiago_public_ws/src/manipulation_challenge/src/perception_module/utils/l2_encoder.onnx"
DECODER_VITSAM_PATH = "/tiago_public_ws/src/manipulation_challenge/src/perception_module/utils/l2_decoder.onnx"

def depth_image_to_point_cloud(depth_image, camera_intrinsics):
    height, width = depth_image.shape
    
    v, u = np.indices((height, width))

    x = (u - camera_intrinsics[0, 2]) * depth_image / camera_intrinsics[0, 0]
    y = (v - camera_intrinsics[1, 2]) * depth_image / camera_intrinsics[1, 1]
    z = depth_image

    points = np.dstack((x, y, z)).reshape(-1, 3)

    return points

def acquire_image(timeout=2.0):
    """
    Acquire RGB image, depth image, and camera info using wait_for_message.
    
    Args:
        timeout: Timeout in seconds for waiting for messages
        
    Returns:
        tuple: (rgb_image_cv2, point_cloud_o3d) or (None, None) on failure
    """
    try:
        rospy.loginfo("[acquire_image] Waiting for RGB image...")
        msg_rgb = rospy.wait_for_message('/xtion/rgb/image_rect_color', Image, timeout=timeout)
        img = bridge.imgmsg_to_cv2(msg_rgb, "bgr8")
        
        rospy.loginfo("[acquire_image] Waiting for depth image...")
        msg_depth = rospy.wait_for_message('/xtion/depth_registered/image_raw', Image, timeout=timeout)
        img_depth = bridge.imgmsg_to_cv2(msg_depth)
        depth_image = np.asarray(img_depth)
        
        rospy.loginfo("[acquire_image] Waiting for camera info...")
        camera_info = rospy.wait_for_message('/xtion/depth_registered/camera_info', CameraInfo, timeout=timeout)
        
        # Extract camera intrinsics
        proj_matrix = camera_info.K
        fx = proj_matrix[0]
        fy = proj_matrix[4]
        cx = proj_matrix[2]
        cy = proj_matrix[5]
        camera_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        # Create point cloud
        point_cloud = depth_image_to_point_cloud(depth_image, camera_intrinsics)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.transform(np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]))
        
        rospy.loginfo("[acquire_image] \u2713 Image acquisition complete")
        return img, pcd
        
    except rospy.ROSException as e:
        rospy.logerr(f"[acquire_image] Timeout waiting for messages: {e}")
        return None, None
    except Exception as e:
        rospy.logerr(f"[acquire_image] Error: {e}")
        return None, None

def get_latest_rgb_cv(timeout=2.0):
    """
    Get latest RGB image as cv2 BGR numpy array.
    
    Args:
        timeout: Timeout in seconds
        
    Returns:
        numpy.ndarray: BGR image or None on failure
    """
    try:
        msg = rospy.wait_for_message('/xtion/rgb/image_rect_color', Image, timeout=timeout)
        return bridge.imgmsg_to_cv2(msg, 'bgr8')
    except Exception as e:
        rospy.logerr(f"[get_latest_rgb_cv] Error: {e}")
        return None

def get_latest_depth_array(timeout=2.0):
    """
    Get latest depth image as numpy array (meters).
    
    Args:
        timeout: Timeout in seconds
        
    Returns:
        numpy.ndarray: Depth array in meters or None on failure
    """
    try:
        msg = rospy.wait_for_message('/xtion/depth_registered/image_raw', Image, timeout=timeout)
        img_depth = bridge.imgmsg_to_cv2(msg)
        depth_array = np.asarray(img_depth).astype(float)
        
        # Convert to meters if needed
        if depth_array.max() > 20.0:
            depth_array = depth_array / 1000.0
            
        return depth_array
    except Exception as e:
        rospy.logerr(f"[get_latest_depth_array] Error: {e}")
        return None

def get_latest_camera_info(timeout=2.0):
    """
    Get latest CameraInfo message.
    
    Args:
        timeout: Timeout in seconds
        
    Returns:
        CameraInfo: Camera info message or None on failure
    """
    try:
        return rospy.wait_for_message('/xtion/depth_registered/camera_info', CameraInfo, timeout=timeout)
    except Exception as e:
        rospy.logerr(f"[get_latest_camera_info] Error: {e}")
        return None

def get_synchronized_frame(timeout=2.0):
    """
    Get synchronized RGB, depth, and camera info.
    
    Args:
        timeout: Timeout in seconds
        
    Returns:
        tuple: (rgb_bgr, depth_array_meters, camera_info) or (None, None, None) on failure
    """
    try:
        rospy.loginfo("[get_synchronized_frame] Acquiring synchronized frame...")
        
        rgb = get_latest_rgb_cv(timeout)
        depth = get_latest_depth_array(timeout)
        cam_info = get_latest_camera_info(timeout)
        
        if rgb is not None and depth is not None and cam_info is not None:
            rospy.loginfo("[get_synchronized_frame] \u2713 Frame synchronized")
            return (rgb, depth, cam_info)
        else:
            rospy.logerr("[get_synchronized_frame] Failed to get all components")
            return (None, None, None)
            
    except Exception as e:
        rospy.logerr(f"[get_synchronized_frame] Error: {e}")
        return (None, None, None)

def local_acquire_image(path):
    """
    Load image and point cloud from local files.
    
    Args:
        path: Path to directory containing scan.jpg and depth_pointcloud.pcd
        
    Returns:
        tuple: (rgb_image_cv2, point_cloud_o3d)
    """
    img_path = path + "scan.jpg"
    depth_path = path + "depth_pointcloud.pcd"

    img = cv2.imread(img_path)
    depth_image = o3d.io.read_point_cloud(depth_path)
    depth_image.transform(np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]))

    return img, depth_image