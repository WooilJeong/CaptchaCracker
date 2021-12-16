import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CaptchaCracker",
    version="0.0.1",
    license='MIT',
    author="Wooil Jeong",
    author_email="wooil@kakao.com",
    description="CaptchaCracker",
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