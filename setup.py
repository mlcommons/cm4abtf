# setup.py
from setuptools import setup
from setuptools._distutils.dist import Distribution
from setuptools.command.install import install
import subprocess
import sys
import importlib.util
import platform
import os

class CustomInstallCommand(install):
    def run(self):

        # Call the standard run method
        install.run(self)

        # Call the custom function
        return self.custom_function()

    def custom_function(self):
        import cmind
        r = cmind.access({'action':'pull', 'automation':'repo', 'artifact':'mlcommons@cm4abtf', 'branch': 'main'})
        print(r)
        if r['return'] > 0:
           return r['return']
    

setup(
    name='cm4abtf',
    long_description='CM scripts to run MLCommons Automotive benchmarks',
    long_description_content_type='text/x-rst',
    version='0.1',
    packages=[],
    install_requires=[
        "setuptools>=60",
        "wheel",
        "cm4mlops",
        "giturlparse",
        "requests",
        "pyyaml"
        ],
    cmdclass={
        'install': CustomInstallCommand,
    },
)
