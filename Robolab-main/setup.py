from setuptools import setup, find_packages


from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name="robolab",
    version="0.1",
    description='Robolab: Robotics Sub-package for Rofunc',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Junjia Liu",
    author_email="jjliu@mae.cuhk.edu.hk",
    url='https://github.com/Skylark0924/Robolab',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['numpy',
                      "trimesh",
                      "mujoco",
                      "mesh_to_sdf",
                      "tqdm",
                      "pycollada",
                      "pyglet==1.5.29",
                      'pytorch_kinematics'],
    python_requires=">=3.7,<3.11",
    keywords=['robotics', 'robot learning', 'learning from demonstration', 'reinforcement learning',
              'robot manipulation'],
    license='MIT',
    entry_points={
        'console_scripts': [
            'rf=rofunc._main:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
