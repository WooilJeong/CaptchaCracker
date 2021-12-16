# CaptchaCracker

![](https://img.shields.io/badge/TensorFlow-2.5.0-red.svg)
![](https://img.shields.io/badge/NumPy-1.19.5-blue.svg)
![](https://img.shields.io/badge/matplotlib-3.5.1-yellow.svg)
[![Linkedin Badge](https://img.shields.io/badge/-WooilJeong-blue?style=plastic&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/wooil/)](https://www.linkedin.com/in/wooil/) 

[한국어 문서](./README-ko.md)

## Introduction

CaptchaCracker is an open source Python library that provides functions to create and apply deep learning models for Captcha Image recognition. You can create a deep learning model that recognizes numbers in the Captcha Image as shown below and outputs a string of numbers, or you can try the model yourself.


### Input

![png](./assets/example01.png)


### Output

```
023062
```


## Installation

```bash
pip install CaptchaCracker
```


## Examples

- Before execution, training data image files in which the actual value of the Captcha image is indicated in the following file names should be prepared.

![png](./assets/example02.png)


### Train and save the model

```python
import glob
from CaptchaCracker import CreateModel

train_img_path = glob.glob("../data/train_numbers_only/*.png")

CM = CreateModel(train_img_path)
model = CM.train_model(epochs=100)
model.save_weights("../model/weights.h5")

```

### Load a saved model to make predictions

```python
from CaptchaCracker import ApplyModel

target_img_path = "../data/target.png"

AM = ApplyModel(target_img_path)
AM.load_saved_weights("../model/weights.h5")

pred = AM.predict()

print(pred)
```


## References

- https://keras.io/examples/vision/captcha_ocr/