from setuptools import find_packages, setup

setup(
    name="tactile_grasp",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "lerobot>=0.4.0",
        "torch>=2.0",
        "numpy",
        "scipy",
        "mujoco>=3.0",
        "gymnasium>=0.28",
        "transforms3d",
    ],
    author="Quinn Ma",
    license="MIT",
)
