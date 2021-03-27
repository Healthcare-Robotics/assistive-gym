from setuptools import setup, find_packages
import sys, os.path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'assistive_gym'))

# with open("README.md", "r") as f:
#     long_description = f.read()

directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assistive_gym', 'envs', 'assets')
data_files = [os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assistive_gym', 'config.ini')]

for root, dirs, files in os.walk(directory):
    for fn in files:
        data_files.append(os.path.join(root, fn))

setup(name='assistive-gym',
    version='1.0',
    packages=find_packages(),
    python_requires='>=3',
    # install_requires=['gym>=0.2.3', 'pybullet', 'numpy', 'keras==2.3.0', 'tensorflow==1.14.0', 'h5py==2.10.0', 'smplx', 'trimesh'] + ['screeninfo==0.6.1' if sys.version_info >= (3, 6) else 'screeninfo==0.2'],
    install_requires=['gym>=0.2.3', 'pybullet @ git+https://github.com/Zackory/bullet3.git#egg=pybullet', 'numpy', 'keras==2.3.0', 'tensorflow==1.14.0', 'h5py==2.10.0', 'smplx', 'trimesh', 'ray[rllib]', 'numpngw', 'tensorflow-probability==0.7.0'] + ['screeninfo==0.6.1' if sys.version_info >= (3, 6) else 'screeninfo==0.2'],
    # description='Physics simulation for assistive robotics and human-robot interaction.',
    # long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Healthcare-Robotics/assistive-gym',
    author='Zackory Erickson',
    author_email='zackory@gatech.edu',
    license='MIT',
    platforms='any',
    keywords=['robotics', 'assitive robotics', 'human-robot interaction', 'physics simulation'],
    package_data={'assistive_gym': data_files},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Microsoft :: Windows', 'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS', 'Intended Audience :: Science/Research',
        "Programming Language :: Python",
        'Programming Language :: Python :: 3.6', 'Topic :: Games/Entertainment :: Simulation',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Framework :: Robot Framework'
    ],
)
