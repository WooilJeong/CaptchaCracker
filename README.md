# CaptchaCracker

![](https://img.shields.io/badge/TensorFlow-2.5.0-red.svg)
![](https://img.shields.io/badge/NumPy-1.19.5-blue.svg)
[![Linkedin Badge](https://img.shields.io/badge/-WooilJeong-blue?style=plastic&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/wooil/)](https://www.linkedin.com/in/wooil/) 

[한국어 문서](https://github.com/WooilJeong/CaptchaCracker/blob/main/README-ko.md)

## Introduction

CaptchaCracker is an open source Python library that provides functions to create and apply deep learning models for Captcha Image recognition. You can create a deep learning model that recognizes numbers in the Captcha Image as shown below and outputs a string of numbers, or you can try the model yourself.


### Input

![png](https://github.com/WooilJeong/CaptchaCracker/raw/main/assets/example01.png)


### Output

```
023062
```


## Installation

```bash
pip install CaptchaCracker
```

## Dependency

```
pip install numpy==1.19.5 tensorflow==2.5.0
```

## Examples

### Train and save the model

Before executing model training, training data image files in which the actual value of the Captcha image is indicated in the file name should be prepared as shown below.

![png](https://github.com/WooilJeong/CaptchaCracker/raw/main/assets/example02.png)


```python
import glob
from CaptchaCracker import CreateModel

train_img_path_list = glob.glob("../data/train_numbers_only/*.png")

CM = CreateModel(train_img_path_list)
model = CM.train_model(epochs=100)
model.save_weights("../model/weights.h5")

```

### Load a saved model to make predictions

```python
from CaptchaCracker import ApplyModel

weights_path = "../model/weights.h5"
AM = ApplyModel(weights_path)

target_img_path = "../data/target.png"
pred = AM.predict(target_img_path)
print(pred)
```


## References

- https://keras.io/examples/vision/captcha_ocr/