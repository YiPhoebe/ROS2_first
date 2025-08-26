from setuptools import setup

package_name = 'image_dir_pub_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='YuJung',
    maintainer_email='you@example.com',
    description='Publish images from a directory to /image_raw',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'image_dir_pub = image_dir_pub_py.main:main',
        ],
    },
)