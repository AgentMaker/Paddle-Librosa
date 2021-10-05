import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="paddlelibrosa",
    version="0.0.1",
    description="Paddle implemention of part of librosa functions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/agentmaker/paddlelibrosa",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'librosa>=0.7.0'
    ],
    python_requires='>=3.6',
)
