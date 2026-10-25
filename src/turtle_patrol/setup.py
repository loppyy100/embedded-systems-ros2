from setuptools import setup

package_name = "turtle_patrol"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="student",
    maintainer_email="student@example.com",
    description="Lab 1 question 1 - parametric turtlesim trajectories (square, star, spiral, flower).",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "patrol = turtle_patrol.patrol_node:main",
        ],
    },
)
