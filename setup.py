from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name="forest", # Replace with your own username
    packages = ['forest'],
    version="0.1",
    author="ThePyProgrammer",
    author_email="prannayagupta@gmail.com",
    description="A data science and machine learning library made by me using codes found online. It is based on the pyforest module.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/ThePyProgrammer/forest",
    download_url = 'https://github.com/ThePyProgrammer/forest/archive/v_01.tar.gz', 
    install_requires=['pandas', 'numpy', 'np'], #external packages as dependencies
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
