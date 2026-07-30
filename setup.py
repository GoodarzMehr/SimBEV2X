'''
SimBEV2X: Scalable Vehicle-to-Everything Data Generation Tool for Multi-Task
Cooperative Perception

Copyright © 2026 Goodarz Mehr
'''

from setuptools import setup, find_packages

# Read long description from README.
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

# Read requirements.
with open('requirements.txt', 'r', encoding='utf-8') as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name='simbev2x',
    version='1.0.0',
    author='Goodarz Mehr',
    author_email='goodarzm@vt.edu',
    description='Scalable Vehicle-to-Everything Data Generation Tool for Multi-Task Cooperative Perception',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/GoodarzMehr/SimBEV2X',
    project_urls={
        'Bug Tracker': 'https://github.com/GoodarzMehr/SimBEV2X/issues',
        'Documentation': 'https://simbev2x.org',
        'Paper': 'https://arxiv.org/abs/2607.23910',
        'Dataset': 'https://drive.google.com/drive/folders/1HVlrp_SrEdSbzj8-BLEFdNtcrp03bXGr?usp=sharing'
    },
    packages=find_packages(exclude=['configs', 'assets']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Operating System :: POSIX :: Linux',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'simbev2x=simbev2x.simbev2x:entry',
            'simbev2x-postprocess=simbev2x_tools.post_processing:entry',
            'simbev2x-visualize=simbev2x_tools.visualization:entry',
        ],
    }
)
