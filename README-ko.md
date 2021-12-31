# CaptchaCracker

![](https://img.shields.io/badge/TensorFlow-2.5.0-red.svg)
![](https://img.shields.io/badge/NumPy-1.19.5-blue.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FWooilJeong%2FCaptchaCracker&count_bg=%2379C83D&title_bg=%23555555&icon=github.svg&icon_color=%23FFFFFF&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![Linkedin Badge](https://img.shields.io/badge/-WooilJeong-blue?style=plastic&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/wooil/)](https://www.linkedin.com/in/wooil/) 

[English document](https://github.com/WooilJeong/CaptchaCracker/blob/main/README.md)  

[Python Captcha 인식 모델 만들기](https://wooiljeong.github.io/python/captcha-cracker/)  


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
import CaptchaCracker as cc

# 학습 이미지 데이터 경로
train_img_path_list = glob.glob("../data/train_numbers_only/*.png")

# 학습 이미지 데이터 크기
img_width = 200
img_height = 50

# 모델 생성 인스턴스
CM = cc.CreateModel(train_img_path_list, img_width, img_height)

# 모델 학습
model = CM.train_model(epochs=100)

# 모델이 학습한 가중치 파일로 저장
model.save_weights("../model/weights.h5")
```

### 저장된 모델 불러와서 예측하기

```python
import CaptchaCracker as cc

# 타겟 이미지 크기
img_width = 200
img_height = 50
# 타겟 이미지 라벨 길이
max_length = 6
# 타겟 이미지 라벨 구성요소
characters = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

# 모델 가중치 파일 경로
weights_path = "../model/weights.h5"
# 모델 적용 인스턴스
AM = cc.ApplyModel(weights_path, img_width, img_height, max_length, characters)

# 타겟 이미지 경로
target_img_path = "../data/target.png"

# 예측값
pred = AM.predict(target_img_path)
print(pred)
```

## 참고

- https://keras.io/examples/vision/captcha_ocr/