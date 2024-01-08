<div align=center>

</div> 

# CV_09조 구해조

<div align=center>

|<img src="https://user-images.githubusercontent.com/72690566/200118081-7f8e4279-04ef-4269-abde-80b9ea89e87a.png" width="80">|<img src="https://user-images.githubusercontent.com/72690566/200118119-d21769d2-ff0d-4e15-9e6d-aa863e700f36.png" width="80">|<img src="https://user-images.githubusercontent.com/72690566/200118141-2de150f1-98cb-4cbd-8ce8-419c1ebb0678.png" width="80">|<img src="https://user-images.githubusercontent.com/72690566/200118162-f25ae93e-18c1-462f-8298-c6ff5c95ee79.png" width="80">|<img src="https://user-images.githubusercontent.com/72690566/200118175-ba5859db-5a2f-4457-a8e2-878f8cc1140e.png" width="80">|
|:---:|:---:|:---:|:---:|:---:|
|김민윤|배종욱|신호준|최수진|전병관|
|T6017|T6071|T6091|T6174|T6152|

</div>

# Result
- Public 2등 -> Private 2등
![image](주소입력필요)



# ⭐DEMO

> 모델 설명을 위한 데모입니다. 해당 파일은 깃헙에 포함되어 있지 않습니다.

### 사진을 입력받아, 마스크 착용 여부 / 성별 여부 / 나이 여부를 구분하는 모델

<img src = "주소입력필요">


# 🌳 Folder Structure
```
.
├── 구해조 입력 필요.

```


# ❓ Team

## 1. 작업 환경

- 컴퓨팅 환경 : V100 GPU
- 협업 도구 : Notion, Slack, Wandb, GitHub, Discord

## 2. 작업의 순서

<div align=center>

<img src="https://user-images.githubusercontent.com/72690566/200120015-b52eb581-764f-41b0-80fe-b083d9accd0f.png">

</div>
  
강의자료에 주어진 Workflow를 참고하여, 프로젝트 타임라인을 위와 같이 설정하였다.

# ❇️ 프로젝트 팀 구성 및 역할

<div align=center>

|전체|문제 정의, 계획 및 타임라인 수립, 모델 튜닝, 아이디어 제시|
|:----------:|:------:|
|김민윤 &nbsp;&nbsp;&nbsp;&nbsp;|metric 구현, validation dataset 구축, background removal & face zoom in, age remove 함수 구현|
|배종욱|Age transformation, Masked Image generation, Re-labeling 툴 제작|
|신호준|BaseLine 코드 제작, Metric 구현, Activation Map 구현, 검수툴 제작, background removal & face zoom in, Backbone 및 Loss 실험|
|전병관|Incorrect Masked Image generation, Dataset grouping|
|최수진|Multi-task Modeling 구현, Backbone & Method Application 실험 및 결과 분석|

</div>
</div>


# **Problem definition**

## 1. 대회목적

주로 비말에서 전파되는 COVID-19를 예방하기위해 올바른 마스크 착용이 중요하다. 그러나 모든 사람의 마스크 착용 상태를 확인하는 것은 비용이 너무 많이 든다. 따라서 본 대회의 목적은 이미지에서 사람의 성별, 나이의 구간, 마스크의 올바른 착용여부를 판단하고자한다.  
**총 18개의 클래스를 예측하는 모델을 제작했다.**

## 2. EDA

### Problem 1. Data imbalance

<div align=center>

<img src="이미지 업로드 해야함">

</div>

 30대부터 40대까지의 데이터의 수량이 매우 부족함을 알 수 있다.  
 클래스 분류는 60대 이상으로 되어있으나 데이터는 60세 데이터만 존재함을 알 수 있다.

- **전략 1.** 외부 데이터 사용이 불가하므로 기존 데이터를 활용해서 새로운 연령대의 이미지 데이터를 생성하자.
- **전략 2.** 기존 데이터셋의 구조에 맞춰서 학습시키기 위해 전략1에서 생성된 이미지에 마스크착용한 이미지를 새롭게 생성하자.
- **전략 3.** Train & Valid에 균등하게 분배될 수 있게하자

### Problem 2. Data mislabel

<div align=center>

<img src="이미지 업로드 해야함">

</div>

- 저작권법에 의해 구체적인 사진을 통한 예시를 제공할 수 없으나, 남성을 여성으로 표현하는 등의 각 클래스(성별, 나이, 마스크착용여부)의 표기가 잘못된 경우를 발견했다.

- **전략 4.** 전수조사를 통해 Mislabel 이미지를 찾아 수정하자.

### Problem 3. Activation Map

<div align=center>

<img src="이미지 입력 필요">

</div>

  성별, 나이, 마스크 착용여부는 모두 얼굴에서 확인한다고 판단했고 제작한 학습된 모델이 이미지를 판단할때 무엇을 보는지 확인해보니 얼굴이 아닌 다른 부분에 더 많은 집중을 하는것을 확인했다.  
  이에 따라 다음과 같은 새로운 전략을 수립하였다.
  
  - **전략 5.** 배경을 제거한 이미지를 학습시켜서 모델이 얼굴에 집중하도록 하자.
  - **전략 6.** 전략 5로 해결이 되지 않는다면 얼굴만 학습하도록 하자.

## 4. Data Preprocessing
Solution은 위의 Problem이나 전략의 순서가 아닌 진행한 작업의 시간순서대로 기술되어 있음.

### Solution 1. 자체 전처리 프로그램 제작
<div align=center>

<img src="이미지 업로드 해야함">

</div>
- 전수조사를 위해 자체제작 툴을 제작하기로 한다. (전략 4)

</div>

### Solution 2. Data Generation

<div align=center>

<img src="이미지 업로드 해야함">

</div>

####	Aging Image Generation  
제공받은 Train에서 Normal사진을 이용해서 하나의 이미지당 10살단위로 20대부터 70대까지 6개의 새로운 연령대의 이미지를 생성해내었다. (전략1)
####	Masked and Incorrect Masked Image Generation  
위에서 생성한 이미지에 서로다른 마스크를 착용한 5장의 Mask 이미지와 1장의 Incorrect 이미지를 생성해내었다. (전략2)

### Solution 3. Image customize

####	Background remove  
<div align=center>

<img src="이미지 업로드 해야함">

</div>

U2-Net 기반의 Rembg 모델을 사용해 Background Removal을 진행했다.


####	Face Zoom in
####	CenterCrop

<div align=center>

<img src="이미지 업로드 해야함">

</div>

입력받은 이미지의 가운데만 추출해봤으나 얼굴의 위치가 이미지별로 모두 상이하였다.
####	MTCNN  

<div align=center>

<img src="이미지 업로드 해야함">

</div>
얼굴과 그 주변부 정보를 남겨, 얼굴의 크기를 일관성있게 유지하고 배경의 노이즈를 제거하고자 했다. 해당 방법이 얼굴을 제대로 탐지하지 못해내었다면 여기서 다시 CenterCrop을 수행하였다.

# 실험 결과 비교

<div align=center>

<img src="이미지 입력 필요">

</div>

</div>

# 🐣 Main strategy

## [1] DataSet & DataLoader 

### Validation Dataset
train과 validation의 분류를 과적합을 방지하기위해 사용된 인물이 중복되지 않고 클래스간 비율이 비슷하게 구현했다.(전략3)
### WeightedRandomSampler  
데이터가 불균형으로 인해 배치에 들어가는 클래스간 비율이 균등하지 않는 문제를 해결하기 위하여 가중치를 클래스별로 다르게 부여하여 배치내의 클래스 수가 비슷하게 유지될 수 있도록한다.
### Age_removal 
실험 도중 F1-score를 확인해보니 모델이 나이클래스에서 잘 학습하지 못하는 이유에 대한 문제를 해결하기 위해 57세이상 60세미만의 데이터를 제외하고 학습을 시도했다.   

## [2] Loss Function
### Cross Entropy
일반적으로 분류문제에 많이 쓰이는 크로스엔트로피 함수를 사용해서 학습을 진행하였다.
### Focal  
학습결과 모델이 Age클래스의 정답률이 낮았고 이를 어려운 문제로 규정 Focal 함수를 사용해서 극복하고자함
## [3] Multi-Task Learning (MTL)


<div align=center>  

![Footer](https://capsule-render.vercel.app/api?type=waving&color=7F7FD5&fontColor=FFFFFF&height=200&section=footer)

</div>
