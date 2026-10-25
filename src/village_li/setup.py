from setuptools import setup

package_name = "village_li"

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
    description="Lab 2 section 2.4 - first Python node (hello) and a simple topic writer.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "hello_node = village_li.hello:main",
            "li_si = village_li.li_si_writer:main",
        ],
    },
)
