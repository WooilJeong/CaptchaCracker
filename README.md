<div align="center">

<b>Open source Python Deep Learning low-code library for generating captcha image recognition models</b><br>
<b>🚀`pip install CaptchaCracker --upgrade`</b>


[![PyPI Latest Release](https://img.shields.io/pypi/v/captchacracker.svg)](https://pypi.org/project/captchacracker/)
![](https://img.shields.io/badge/TensorFlow-2.5.0-red.svg)
![](https://img.shields.io/badge/NumPy-1.19.5-blue.svg)

[한국어 문서](https://github.com/WooilJeong/CaptchaCracker/blob/main/README-ko.md)

<div align="left">


<br>

## CaptchaCracker

CaptchaCracker is an open source Python library that provides functions to create and apply deep learning models for Captcha Image recognition. You can create a deep learning model that recognizes numbers in the Captcha Image as shown below and outputs a string of numbers, or you can try the model yourself.


### Input

![png](https://github.com/WooilJeong/CaptchaCracker/raw/main/assets/example01.png)


### Output

```
023062
```

## Web Demo

Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/CaptchaCracker)


<br>

## Installation

```bash
pip install CaptchaCracker
```

<br>

## Dependency

```
pip install numpy==1.19.5 tensorflow==2.5.0
```

<br>

## Examples

### Train and save the model

Before executing model training, training data image files in which the actual value of the Captcha image is indicated in the file name should be prepared as shown below.

- [Download Sample Dataset](https://github.com/WooilJeong/CaptchaCracker/raw/main/sample.zip)

![png](https://github.com/WooilJeong/CaptchaCracker/raw/main/assets/example02.png)


```python
import glob
import CaptchaCracker as cc

# Training image data path
train_img_path_list = glob.glob("../data/train_numbers_only/*.png")

# Training image data size
img_width = 200
img_height = 50

# Creating an instance that creates a model
CM = cc.CreateModel(train_img_path_list, img_width, img_height)

# Performing model training
model = CM.train_model(epochs=100)

# Saving the weights learned by the model to a file
model.save_weights("../model/weights.h5")
```

### Load a saved model to make predictions

```python
import CaptchaCracker as cc

# Target image data size
img_width = 200
img_height = 50
# Target image label length
max_length = 6
# Target image label component
characters = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

# Model weight file path
weights_path = "../model/weights.h5"
# Creating a model application instance
AM = cc.ApplyModel(weights_path, img_width, img_height, max_length, characters)

# Target image path
target_img_path = "../data/target.png"

# Predicted value
pred = AM.predict(target_img_path)
print(pred)
```


<br>

## References

- https://keras.io/examples/vision/captcha_ocr/

<br>

## Contributors

<a href="https://github.com/wooiljeong/captchacracker/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=wooiljeong/captchacracker" />
</a>

<br>

<div align=center>

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FWooilJeong%2FCaptchaCracker&count_bg=%2379C83D&title_bg=%23555555&icon=github.svg&icon_color=%23FFFFFF&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

</div>