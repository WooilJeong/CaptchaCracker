import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CaptchaCracker",
    version="0.0.3",
    license='MIT',
    author="Wooil Jeong",
    author_email="wooil@kakao.com",
    description="CaptchaCracker is an open source Python library that provides functions to create and apply deep learning models for Captcha Image recognition.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WooilJeong/CaptchaCracker",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)