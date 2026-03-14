from setuptools import find_packages, setup
from glob import glob
import os

package_name = "carters_goal"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="isaac sim",
    maintainer_email="isaac-sim@todo.todo",
    description="Package to set goals for navigation stack.",
    license="NVIDIA Isaac ROS Software License",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "SetNavigationGoal = carters_goal.set_goal:main",
            "MapfGoalPublisher = carters_goal.mapf_goal_publisher:main",
            "NamespacedTfBridge = carters_goal.namespaced_tf_bridge:main",
            "InitialPoseTfPublisher = carters_goal.initial_pose_tf_publisher:main",
            "MapfNav2Executor = carters_goal.mapf_nav2_executor:main",
        ]
    },
)
