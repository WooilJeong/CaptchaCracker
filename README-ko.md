# CaptchaCracker

![](https://img.shields.io/badge/TensorFlow-2.5.0-red.svg)
![](https://img.shields.io/badge/NumPy-1.19.5-blue.svg)
[![Linkedin Badge](https://img.shields.io/badge/-WooilJeong-blue?style=plastic&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/wooil/)](https://www.linkedin.com/in/wooil/) 

[English document](https://github.com/WooilJeong/CaptchaCracker/blob/main/README.md)

## 소개

CaptchaCracker는 Captcha Image 인식을 위한 딥 러닝 모델 생성 기능과 적용 기능을 제공하는 오픈소스 파이썬 라이브러리입니다. 아래와 같은 Captcha Image의 숫자를 인식해 숫자 문자열을 출력하는 딥 러닝 모델을 만들거나 모델을 직접 사용해볼 수 있습니다.


### 입력 이미지

![png](https://github.com/WooilJeong/CaptchaCracker/raw/main/assets/example01.png)


### 출력 문자열

```
023062
```


## 설치

```bash
pip install CaptchaCracker
```

## 의존성

```
pip install numpy==1.19.5 tensorflow==2.5.0
```

## 예제

### 모델 학습 및 저장하기

모델 학습 실행에 앞서 아래와 같이 파일명에 Captcha 이미지의 실제값이 표기된 학습 데이터 이미지 파일들이 준비되어 있어야 합니다.

- [샘플 데이터 다운로드](https://github.com/WooilJeong/CaptchaCracker/raw/main/sample.zip)

![png](https://github.com/WooilJeong/CaptchaCracker/raw/main/assets/example02.png)


```python
import glob
from CaptchaCracker import CreateModel

train_img_path_list = glob.glob("../data/train_numbers_only/*.png")

CM = CreateModel(train_img_path_list)
model = CM.train_model(epochs=100)
model.save_weights("../model/weights.h5")

```

### 저장된 모델 불러와서 예측하기

```python
from CaptchaCracker import ApplyModel

weights_path = "../model/weights.h5"
AM = ApplyModel(weights_path)

target_img_path = "../data/target.png"
pred = AM.predict(target_img_path)
print(pred)
```

## 참고

- https://keras.io/examples/vision/captcha_ocr/