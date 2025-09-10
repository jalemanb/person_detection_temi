import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    rviz_config_file = os.path.join(
        get_package_share_directory('person_detection_ros'), 'rviz', 'config.rviz'
    )

    use_sim_time = LaunchConfiguration('use_sim_time')

    namespace = LaunchConfiguration('namespace')
    use_rviz = LaunchConfiguration('use_rviz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (rosbag) clock if true'
        ),

        DeclareLaunchArgument(
            'namespace',
            default_value='',
            description='Namespace to push topics into'
        ),

        DeclareLaunchArgument(
            'use_rviz',
            default_value='false',
            description='Whether to launch RViz2'
        ),

        # Person detection node
        Node(
            package='person_detection_ros',
            executable='person_detection_node',
            name='person_detection_node',
            namespace=namespace,
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}]
        ),

        # RViz2 node (only if launch_rviz == true)
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            namespace=namespace,
            output='screen',
            arguments=['-d', rviz_config_file],
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(use_rviz)
        ),
    ])