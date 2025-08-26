import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'yolo_subscriber_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(
        include=(package_name, package_name + '.*')),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_sub = yolo_subscriber_py.main:main',
            'yolo_subscriber = yolo_subscriber_py.main:main',
            'yolo_subscriber_py_node = yolo_subscriber_py.main:main',
        ],
    },
)
