from setuptools import setup, find_packages

setup(
    name="neuroscience_tools",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "zstandard",
        "scipy",
        "PyWavelets",
        "tqdm",
        "scipy",
        "matplotlib",
        'jax',
        'jaxlib',
    ],
    author="Mojtaba",
    author_email="mojtaba.alavi@isc.cnrs.fr",
    description="A Python package for handling neuroscience data with Zstandard compression.",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
)
