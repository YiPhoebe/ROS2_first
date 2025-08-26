from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    image_pub = Node(
        package='image_pub_cpp',
        executable='image_publisher_node',
        name='image_pub',
        parameters=[{
            'dir': '/workspace/dataset/bdd100k/bdd100k/images/100k/val',
            'fps': 1.5,
            'frame_id': 'camera_frame'
        }]
    )

    yolo = Node(
        package='yolo_subscriber_py',
        executable='yolo_sub',
        name='yolo_subscriber_py_node',
        parameters=[{
            'weights': '/workspace/yolov8n.pt',
            'conf': 0.15,
            'imgsz': 640,
            'every_n': 1
        }],
        output='screen'
    )

    return LaunchDescription([image_pub, yolo])
