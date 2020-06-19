from setuptools import setup, find_packages


with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()


setup(
    name="clearbox-wrapper",
    version="0.0.1",
    author="Clearbox AI",
    author_email="info@clearbox.ai",
    description="An agnostic wrapper for the most common frameworks of ML models",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url="https://github.com/ClearBox-AI/clearbox-wrapper",
    packages=find_packages(include=['clearbox_wrapper', 'clearbox_wrapper.*']),
    install_requires=[
        'cloudpickle'
    ],
    keywords="ML wrapper machine learning model"
)
