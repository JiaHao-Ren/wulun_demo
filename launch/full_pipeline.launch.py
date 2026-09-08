# 视觉链路 + 延迟报告。
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('edge_inference_optimizer')
    pipeline_cfg = os.path.join(pkg_share, 'config', 'pipeline.yaml')
    servo_cfg = os.path.join(pkg_share, 'config', 'servo_mapping.yaml')

    args = [
        DeclareLaunchArgument(
            'model_path',
            description='emotion-ferplus-8.onnx 路径'),
        DeclareLaunchArgument(
            'backend', default_value='cpu',
            description='cpu | cuda'),
        DeclareLaunchArgument(
            'threads', default_value='1',
            description='ONNX 线程数'),
        DeclareLaunchArgument(
            'fps', default_value='30.0'),
        DeclareLaunchArgument(
            'image_dir', default_value='',
            description='图片目录，空则用合成画面'),
        DeclareLaunchArgument(
            'report_period_s', default_value='5.0'),
    ]

    camera = Node(
        package='edge_inference_optimizer',
        executable='camera_node',
        name='camera_node',
        output='screen',
        parameters=[pipeline_cfg, {
            'fps': LaunchConfiguration('fps'),
            'image_dir': LaunchConfiguration('image_dir'),
        }],
    )

    detector = Node(
        package='edge_inference_optimizer',
        executable='emotion_detector_node',
        name='emotion_detector',
        output='screen',
        parameters=[pipeline_cfg, {
            'model_path': LaunchConfiguration('model_path'),
            'backend': LaunchConfiguration('backend'),
            'threads': LaunchConfiguration('threads'),
        }],
    )

    servo = Node(
        package='edge_inference_optimizer',
        executable='servo_mapper_node',
        name='servo_mapper',
        output='screen',
        parameters=[servo_cfg],
    )

    reporter = Node(
        package='edge_inference_optimizer',
        executable='latency_reporter_node',
        name='latency_reporter',
        output='screen',
        parameters=[pipeline_cfg, {
            'report_period_s': LaunchConfiguration('report_period_s'),
        }],
    )

    return LaunchDescription(args + [camera, detector, servo, reporter])
