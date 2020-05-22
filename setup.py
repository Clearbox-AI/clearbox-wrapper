from setuptools import setup, find_packages


with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()


setup(
    name="clearbox-wrapper",
    version="0.0.1",
    author="ClearBox AI",
    author_email="info@clearbox.ai",
    description="An agnostic wrapper for the most common formats of ML models",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT License",
    url="https://github.com/ClearBox-AI/clearbox-wrapper",
    packages=find_packages(),
    install_requires=[
    ],
    keywords="ML wrapper machine learning model"
)
