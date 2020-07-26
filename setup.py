from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name="forest", # Replace with your own username
    version="0.0.1",
    author="ThePyProgrammer",
    author_email="prannayagupta@gmail.com",
    description="A data science and machine learning library made by me using codes found online. It is based on the pyforest module.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ThePyProgrammer/forest",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
