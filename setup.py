from setuptools import setup, find_packages

with open("README.md", "r") as file:
    long_description = file.read()

setup(
    name="anomaly",
    version="0.0.1",
    author="Edgar Ortiz",
    author_email="ed.ortizm@gmail.com",
    packages=find_packages(where="src", include=["[a-z]*"], exclude=[]),
    package_dir={"": "src"},
    description="Python code to compute anomaly scores",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ed-ortizm/xai-astronomy",
    license="MIT",
    keywords="astrophysics, galaxy, Machine Learning, anomaly detection",
)
