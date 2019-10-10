from setuptools import setup, find_packages
import sys, os.path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'assistive-gym'))

with open("README.md", "r") as f:
    long_description = f.read()

directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assistive-gym', 'envs', 'assets')
data_files = [os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assistive-gym', 'config.ini')]

for root, dirs, files in os.walk(directory):
    for fn in files:
        data_files.append(os.path.join(root, fn))

setup(name='assistive-gym',
    version='0.100',
    packages=find_packages(),
    python_requires='>=3',
    install_requires=['gym>=0.2.3', 'pybullet', 'numpy', 'keras', 'tensorflow'],
    description='Physics simulation for assistive robotics and human-robot interaction.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Healthcare-Robotics/assistive-gym',
    author='Zackory Erickson',
    author_email='zackory@gatech.edu',
    license='MIT',
    platforms='any',
    keywords=['robotics', 'assitive robotics', 'human-robot interaction', 'physics simulation'],
    package_data={'assistive-gym': data_files},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: zlib/libpng License',
        'Operating System :: Microsoft :: Windows', 'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS', 'Intended Audience :: Science/Research',
        "Programming Language :: Python",
        'Programming Language :: Python :: 3.4', 'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6', 'Topic :: Games/Entertainment :: Simulation',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Framework :: Robot Framework'
    ],
)
