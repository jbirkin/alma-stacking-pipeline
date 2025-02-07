from setuptools import setup, find_packages

setup(
    name="alma_stacking_pipeline",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "astropy",
        "pyyaml",
        "emcee",
        "tqdm",
        "pyregion"
    ],
    entry_points={
        "console_scripts": [
            "alma-stack=alma_stacking_pipeline.src.main:main",
        ],
    },
)