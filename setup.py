from setuptools import setup, find_packages
import codecs
import os


VERSION = '0.0.1'
DESCRIPTION = 'Linear/Multli Regression Mathematical Function in one line of code'
LONG_DESCRIPTION = 'Just provide x and y list and there you have it the Mathemtical function + accuracy based on the x and y list.'

# Setting up
setup(
    name="sklearn_fluent",
    version=VERSION,
    author="YusiferZendric (Aditya Singh)",
    author_email="<yzendric@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['scikit-learn'],
    keywords=['python', 'sklearn', 'mathematical functions', 'functions', 'linear regressions'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
