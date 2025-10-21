# FilteredObjectPercept message
string type_label
sensor_msgs/PointCloud2 pcl

# Oriented Bounding Box (using PCA)
geometry_msgs/Point bounding_box_center
geometry_msgs/Quaternion bounding_box_orientation
float64[3] bounding_box_dimensions  # [length, width, height] along principal axes

geometry_msgs/Point centroid

# Filtered Oriented Bounding Box
geometry_msgs/Point filtered_bounding_box_center
geometry_msgs/Quaternion filtered_bounding_box_orientation
float64[3] filtered_bounding_box_dimensions

geometry_msgs/Point filtered_pose