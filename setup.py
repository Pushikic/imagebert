import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="imagebert",
    version="0.0.1-snapshot",
    author="S. Maeda",
    description="ImageBERT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maeda6uiui/imagebert",
    packages=setuptools.find_packages("src"),
    package_dir={"":"src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
