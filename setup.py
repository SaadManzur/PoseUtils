from setuptools import setup

setup(
    name='poseutils',
    version='0.1.0',    
    description='A simple package containing common essentials for pose based research',
    author='Saad Manzur',
    author_email='smanzur@uci.edu',
    license='LICENSE.txt',
    packages=['poseutils'],
    install_requires=['numpy', 'tqdm'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)