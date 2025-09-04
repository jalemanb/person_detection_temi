from setuptools import find_packages, setup
from glob import glob
import os
package_name = 'person_detection_ros'
# submodules = package_name + "/submodules"
# utils = package_name + "/submodules" +"/utils"

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[package_name, f'{package_name}.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/detect.launch.py']),
        ('share/' + package_name + '/launch', ['launch/vis.launch.py']),
        ('share/' + package_name + '/models', glob('models/*')),
        ('share/' + package_name + '/template_imgs', glob('template_imgs/*')),
        ('share/' + package_name + '/rviz', glob('rviz/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='enrique',
    maintainer_email='jesus.aleman@uni-bielefeld.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'person_detection_node = person_detection_ros.person_detection_node:main',
            'eval = person_detection_ros.eval:main',
            'ablation = person_detection_ros.ablation:main',
            'video = person_detection_ros.attention_video:main',

        ],
    },
)