"""Start the image source and the recogniser together.

    ros2 launch gesture_recognition gesture.launch.py
    ros2 launch gesture_recognition gesture.launch.py source:=pattern
    ros2 launch gesture_recognition gesture.launch.py use_cam2image:=true
    ros2 launch gesture_recognition gesture.launch.py show_window:=false with_dog:=true

`use_cam2image:=true` uses the stock `image_tools cam2image` node from the lab
guide instead of our own publisher; everything downstream is identical.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    source = LaunchConfiguration("source")
    show_window = LaunchConfiguration("show_window")
    use_cam2image = LaunchConfiguration("use_cam2image")
    with_dog = LaunchConfiguration("with_dog")
    dry_run = LaunchConfiguration("dry_run")

    return LaunchDescription([
        DeclareLaunchArgument("source", default_value="0",
                              description="camera index, video file, or 'pattern'"),
        DeclareLaunchArgument("show_window", default_value="true"),
        DeclareLaunchArgument("use_cam2image", default_value="false",
                              description="use image_tools cam2image instead"),
        DeclareLaunchArgument("with_dog", default_value="false",
                              description="also start the CyberDog bridge"),
        DeclareLaunchArgument("dry_run", default_value="true",
                              description="bridge logs commands instead of sending LCM"),

        Node(package="image_tools", executable="cam2image", name="cam2image",
             output="screen", condition=IfCondition(use_cam2image)),

        Node(package="gesture_recognition", executable="image_source",
             name="image_source", output="screen",
             parameters=[{"source": source}],
             condition=UnlessCondition(use_cam2image)),

        Node(package="gesture_recognition", executable="gesture_node",
             name="gesture_recognition_node", output="screen",
             parameters=[{"show_window": show_window}]),

        Node(package="cyberdog_gesture_bridge", executable="bridge",
             name="cyberdog_gesture_bridge", output="screen",
             parameters=[{"dry_run": dry_run}],
             condition=IfCondition(with_dog)),
    ])
