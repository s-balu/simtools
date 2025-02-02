from setuptools import setup

setup(
    name="simtools",
    version="0.1",
    description="Tools for analysing cosmological simulation outputs",
    url="https://github.com/kriswalker/simtools",
    author="Kris Walker",
    author_email="kris.walker@icrar.org",
    package_dir={"simtools": "./simtools/"},
    packages=["simtools"],
    setup_requires=["numpy"],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "h5py",  #'py-sphviewer',
        "tabulate",
        "scikit-learn",
        "pathos",
        "healpy",
    ],
    include_package_data=True,
    zip_safe=False,
)
