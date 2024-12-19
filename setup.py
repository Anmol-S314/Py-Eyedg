from setuptools import setup, find_packages

setup(
    name="Py-Eyedg",
    version="0.1",
    packages=find_packages(),
    install_requires=[],
    author="Anmol",
    author_email="anmol.shetty@iiitb.ac.in",
    description="A python substitute for the MATLAB Eyediagram function",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Anmol-S314/Py-Eyedg",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)