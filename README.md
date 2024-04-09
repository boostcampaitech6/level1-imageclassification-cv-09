<div align=center>

</div> 

# CV_09조 구해조

<br>


# 🏅 Result
- Public 2등 → Private 2등
![image](https://github.com/boostcampaitech6/level1-imageclassification-cv-09/assets/46400961/c3544348-9eba-495f-b38c-0900559f8c36)
<br>

- Project Wrap-up Report: [Link](https://github.com/boostcampaitech6/level1-imageclassification-cv-09/blob/main/level1_cv-09_wrapup_report_compressed.pdf)


# 🌳 File Structure
```
📦level1-imageclassification-cv-09
 ┣ 📂config # 프로그램 동작 설정 폴더
 ┃ ┣ 📜ensemble.yaml # 앙상블 설정 파일
 ┃ ┣ 📜test.yaml # 추론 설정 파일
 ┃ ┗ 📜train.yaml # 훈련 설정 파일
 ┣ 📂model # 모델 관련 폴더
 ┃ ┣ 📜losses.py # 손실 함수 정의 파일
 ┃ ┣ 📜models.py # 모델 코드 파일
 ┃ ┗ 📜optimizers.py # 최적화 알고리즘 파일
 ┣ 📂modules
 ┃ ┣ 📜datasets.py # 사용자 정의 데이터셋 파일
 ┃ ┣ 📜logger.py # 로깅 파일
 ┃ ┣ 📜metrics.py # 평가 지표 파일
 ┃ ┣ 📜schedulers.py # 학습률 스케쥴러 파일
 ┃ ┣ 📜transforms.py # 데이터 변환 파일
 ┃ ┗ 📜utils.py # 유틸리티 함수 파일
 ┣ 📂tools
 ┃ ┣ 📂activation_map
 ┃ ┃ ┗ 📜activation_map.py # 활성화 맵 생성 도구
 ┃ ┣ 📂background_faceDetection
 ┃ ┃ ┗ 📜bg.py # 배경 얼굴 감지 파일
 ┃ ┣ 📂Generation
 ┃ ┃ ┣ 📜basic_mask.py # 기본 마스크 생성 파일
 ┃ ┃ ┣ 📜Mask_Gen.py # 마스크 생성 파일
 ┃ ┃ ┣ 📜mask_the_face.py # 얼굴에 마스크 적용 파일
 ┃ ┃ ┗ 📜Mask_Gen_Multi-Process.py # 다중 프로세스를 사용한 마스크 생성 파일
 ┃ ┣ 📂incorrect_grouping
 ┃ ┃ ┣ 📜jackup.py # 그룹화 오류 수정 파일
 ┃ ┃ ┣ 📜mask_incorrect_mask.py # 잘못 적용된 마스크 수정 파일
 ┃ ┃ ┗ 📜step1_resize.py # 이미지 크기 조정 파일
 ┃ ┗ 📂labelingTool
 ┃ ┃ ┣ 📜cleaningLabelingTool.py # 라벨링 도구 파일
 ┃ ┃ ┗ 📜Relabeling_Tool.py # 라벨링 재지정 파일
 ┣ 📜multi_test.py # MTL 테스트 실행 파일
 ┣ 📜multi_train.py # MTL 훈련 실행 파일
 ┣ 📜test.py # 테스트 실행 파일
 ┣ 📜test_ensemble.py # 앙상블 테스트 실행 파일
 ┣ 📜train.py # 훈련 실행 파일
 ┗ 📜valid_ensemble.py # 앙상블 검증 실행 파알

```


# 🧑‍🤝‍🧑 Team

## 1. 멤버 

<div align=center>

|<img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-09/assets/136993865/c181bf23-c6a6-42b3-8ec6-415b6ebbf5dc" width="80">|<img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-09/assets/136993865/9ee5e275-eff5-4e39-91c0-2d5d4d98f8e5" width="80">|<img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-09/assets/136993865/be823de7-bf3d-4e37-9789-9592f8d2259d" width="80">|<img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-09/assets/136993865/1b006cf8-9c33-4eef-9801-294a2c9e984c" width="80">|<img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-09/assets/136993865/d003599f-4092-41e2-ba01-c7e5153f5655" width="80">|
|:---:|:---:|:---:|:---:|:---:|
|[김민윤](https://github.com/minyun-e)|[배종욱](https://github.com/Eddie-JUB)|[신호준](https://github.com/internationalwe)|[최수진](https://github.com/sujin-1013)|[전병관](https://github.com/wjsqudrhks)|
|T6017|T6071|T6091|T6174|T6152|

</div>
<br>


## 2. 팀 역할

<div align=center>

|전체|문제 정의, 계획 및 타임라인 수립, 모델 튜닝, 아이디어 제시|
|:-------------------:|:------|
|김민윤&nbsp;&nbsp;&nbsp;&nbsp;|metric 구현, validation dataset 구축, background removal & face zoom in, age remove 함수 구현|
|배종욱&nbsp;&nbsp;&nbsp;&nbsp;|Age transformation, Masked Image generation, Re-labeling 툴 제작|
|신호준&nbsp;&nbsp;&nbsp;&nbsp;|BaseLine 코드 제작, Metric 구현, Activation Map 구현, 검수툴 제작, background removal & face zoom in, Backbone 및 Loss 실험|
|전병관&nbsp;&nbsp;&nbsp;&nbsp;|Incorrect Masked Image generation, Dataset grouping|
|최수진&nbsp;&nbsp;&nbsp;&nbsp;|Multi-task Modeling 구현, Backbone & Method Application 실험 및 결과 분석|

</div>


<br>

## 3. 작업 환경

- 컴퓨팅 환경 : V100 GPU
- 협업 도구 : Notion, Slack, Wandb, GitHub, Discord

<br>
<br>



# ❓ **Problem definition**

## 1. 대회목적

주로 비말에서 전파되는 COVID-19를 예방하기위해 올바른 마스크 착용이 중요하다. 그러나 모든 사람의 마스크 착용 상태를 확인하는 것은 비용이 너무 많이 든다. 따라서 본 대회의 목적은 이미지에서 사람의 성별, 나이의 구간, 마스크의 올바른 착용여부를 판단하고자한다.  
**총 18개의 클래스를 예측하는 모델을 제작했다.**

<br>


## 2. EDA

### Problem 1. Mislabeled Data

<div align=center>

</div>

저작권법에 의해 구체적인 사진을 통한 예시를 제공할 수 없으나, 제공된 데이터셋을 시각화하는 과정에서 남성을 여성으로 표현하는 것과 같이 성별, 나이, 마스크 착용 여부 등의 잘못된 레이블을 다수 발견하였다.

- **전략 1.** 전수조사와 함께 Data Cleansing을 진행하자.

<br>

### Problem 2. Insufficient Focus on the Face

<div align="center" >

<img width="200" alt="Not Background
remove Activation Map" src="https://github.com/boostcampaitech6/level1-imageclassification-cv-09/assets/136993865/b0f5cb04-60e0-4fab-91c5-035ecbd4b605">

< Not Background
remove Activation Map >

</div>

이미지 전처리에 앞서 Activation Map을 통해 모델이 얼굴보다는 옷이나 배경에 더 많이 집중하는 것을 확인했다. 이는 얼굴을 학습하여 클래스를 분류하는 데 있어 noise로 작용할 가능성이 크다고 판단하여 다음과 같은 전략을 세웠다.

  
  - **전략 2.** 모델이 얼굴에 집중할 수 있도록 이미지에서 배경을 제거하자.
  - **전략 3.** **전략 2**로 해결되지 않는다면 Face Zoom in 까지 추가적으로 진행하여 얼굴만 학습하도록 하자.

<br>
    
### Problem 3. Data Imbalance

<div align="center" >

<img width="600" alt="Number of Data per Class"   src="https://github.com/boostcampaitech6/level1-imageclassification-cv-09/assets/136993865/bea72ac2-adb3-491a-be13-4b918c9cf09b">

< Number of Data per Class >
</div>
<br>
    
각 클래스별 학습 데이터의 분포를 정확히 이해하기 위해 전체 데이터 구성을 시각화하여 분석한 결과, 학습 데이터의 클래스별 분포에 불균형이 크게 나타나는 것을 확인하였다.

<br>
<br>

<div align="center" >

<img width="230" alt="Mask Class Distribution"   src="https://github.com/boostcampaitech6/level1-imageclassification-cv-09/assets/136993865/8bb62be0-cee5-49fb-9496-6e2a9f14a048">
        
< Mask Class Distribution >
</div>
<br>

다음은 주어진 데이터셋을 마스크 착용 상태에 따라 나누어 분석하였다. 데이터셋은 한 사람당 마스크 착용(Mask, 5장), 비정상 착용(Incorrect, 1장), 미착용(Normal, 1장) 이미지로 구성되어 있으며, 마스크 착용 상태 별 데이터의 비율은 각각 약 72%, 14%, 14%이다.
        
<br>
<br>

<div align="center" >

<img width="1860" alt="Age Class Distribution" src="https://github.com/boostcampaitech6/level1-imageclassification-cv-09/assets/136993865/81c3321d-d5e9-4405-a12f-1ecdc7243a75">
        
< Age Class Distribution >
</div>
<br>

또한, 나이에 따라 데이터를 분류하여 추가적인 분석을 진행하였다. 30세 미만을 ʻYoung’, 30세 이상 60세 미만을 ʻMiddle’, 그리고 60세 이상을 ʻOld’로 각각 구분하여 살펴본 결과, ʻOld’ 구간의 데이터가 가장 적다는 것을 확인하였다. 이어서 각 연령대별 데이터 분포를 분석하였을 때, 30세와 40대의 데이터는 상대적으로 부족하였고, 70세 이상의 데이터는 존재하지 않았다.

<br>
        
- **전략 4.** 데이터 불균형 문제를 해결하기 위해 마스크를 착용하지 않은 이미지를 활용하여 다양한 연령대의 이미지를 생성하자.
- **전략 5.** 기존 데이터셋의 구조에 맞춰 **전략 4**에서 생성된 이미지에 마스크를 합성하여 학습시키자.
- **전략 6.** 모델의 균형 잡힌 학습을 위해 Train과 Valid 데이터셋에 데이터가 균등하게 분배되도록 조정하자.
<br>
<br>

# ❤️‍🩹 Problem Solve

## Solution 1. Tool for Relabeling
<div align="center" >

<img width="600" alt="Data Error Checking Tool" src="https://github.com/boostcampaitech6/level1-imageclassification-cv-09/assets/136993865/5155cd91-fe3d-46dc-bab4-8a94a442422d">

< Data Error Checking Tool >

</div>

<br>

<div align="center" >

<img width="600" alt="Data Labeling Tool" src="https://github.com/boostcampaitech6/level1-imageclassification-cv-09/assets/136993865/76c939c5-f327-4889-bb3c-03d3a6f68608">

< Data Labeling Tool >

</div>

<br>

효율적인 Data Relabeling 작업을 위해 Data Error Checking Tool과 Data Labeling Tool을 직접 개발하여 활용하였다.
우선 Data Checking Tool을 이용하여 레이블과 이미지가 일치하지 않는 경우를 1차적으로 선별한 후, 팀 회의를 통해 정립된 가이드라인에 따라 Data Labeling Tool로 잘못된 레이블을 수정하였다.
   
<br>


## Solution 2. Data Generation

<div align="center" >

<img width="450" alt="image" src="https://github.com/boostcampaitech6/level1-imageclassification-cv-09/assets/136993865/d2ffd801-091f-4939-b298-d62c9ef6ba84">

</div>
    
    
<br>
<br>

<div align="center" >

<img width="700" alt="Convert face dataset to masked dataset" src="https://github.com/boostcampaitech6/level1-imageclassification-cv-09/assets/136993865/717976ee-9976-4a3e-8b9e-d30550536272">

< Convert face dataset to masked dataset >

</div>
<br>


Old 클래스(60+)가 다른 클래스 대비 상대적으로 데이터가 많이 부족하며, 실험을 통해 해당 클래스의 F1-sccore가 낮게 나오는 것을 확인하였다. 이를 해결하고자 Age transformation, masked, incorrected wearing mask 데이터를 생성하여 전체 데이터셋이 나이대 별로 균형을 이루게 재구성하였다.

- 다양한 연령 이미지를 생성하기 위해 Only a Matter of Style - Age Transformation Using a Style-Based Regression Model (SIGGRAPH 2021)에서 제안한 방법을 사용하였다.
- 마스크 합성 데이터를 생성하기 위해 MaskTheFace에서 제안된 기법을 적용하였다.
<br>

## Solution 3. Image Preprocessing

<div align="center" >

<img width="600" alt="image" src="https://github.com/boostcampaitech6/level1-imageclassification-cv-09/assets/136993865/64fb1fb1-69ca-4e27-9337-b69ce3e1d86a">

</div>
    
    
<br>


<div align="center" >

<img width="600" alt="image" src="https://github.com/boostcampaitech6/level1-imageclassification-cv-09/assets/136993865/11bf152f-6d03-4778-adbb-5fcc88b5d4e7">

</div>
<br>

Activation Map 분석을 통해 모델이 얼굴보다는 옷이나 배경에 더 많이 집중되고 있음을 확인한 후, 모델이 얼굴에 더욱 잘 집중할 수 있도록 이미지에서 배경을 제거하였다. 그 결과 모델이 배경을 지우기 전보다 얼굴에 더 집중하긴 했지만 여전히 옷과 배경에도 집중이 분산되는 것이 확인되어 Face Zoom in 까지 추가적으로 진행했다.

- 이미지의 배경 제거는 딥러닝 기술을 활용하여 U2-Net 기반의 Rembg 모델을 사용했다.
- Face zoom in은 속도, 성능 부분에서 모두 준수하다고 알려진 MTCNN을 사용해 얼굴 검출을 한 후 Crop을 진행했다. 얼굴이 검출되지 않는 경우는 torchvision의 CenterCrop으로 처리했다.

<br>

## Solution 4. Construct Validation Dataset

<div align="center" >

<img width="600" alt="image" src="https://github.com/boostcampaitech6/level1-imageclassification-cv-09/assets/136993865/9ea51ee5-e2be-4cf4-87bf-4807ef208cc0">

</div>
    
    
<br>
<br>

<div align="center" >

<img width="600" alt="image" src="https://github.com/boostcampaitech6/level1-imageclassification-cv-09/assets/136993865/e1200a6c-d3df-4ad7-a14a-c9f9463b57f1">


</div>
    
    
<br>
    
베이스라인 코드로 실험한 결과를 통해 테스트 데이터셋과 검증 데이터셋의 성능 차이가 큼을 확인하였다. 이는 전체 이미지 기준으로 random split 되기 때문에 학습 데이터셋에 있는 사람이 검증 데이터셋에도 포함될 소지가 있어 과적합을 유발할 수 있다고 판단하여 데이터를 profile 기준으로 split 하였다. 또한, 데이터 불균형으로 인해 학습 중 batch에 들어가는 클래스의 개수가 균등하게 들어가지 않는 문제를 해결하기 위해 클래스 별로 가중치를 다르게 부여하였다.

- 데이터를 profile 기준으로 split 하되 sklearn의 stratify를 사용하여 학습과 검증의 각 클래스 비율이 비슷하게 random split 되도록 함수를 구현하였다.
- 클래스 별 가중치는 WeightedRandomSampler를 사용하여 batch 내의 클래스 개수가 비슷하게 유지될 수 있도록 하였다.


<br>
<br>

# ✈️ Model Strategy
## Strategy 1. Baseline Code

## Strategy 2. BackBone Models

<div align="center" >
 
![image](https://github.com/boostcampaitech6/level1-imageclassification-cv-09/assets/46400961/20a6dcbd-adf9-42aa-9f4f-97930f3e9b48)

</div>
   
훈련시킨 Sota급 모델인 Caformer(B36)이 검증(Val) F1-score에서는 높은 성과를 보였으나, 테스트(Test) F1-score에서는 예상치 못하게 크게 하락하는 결과를 보였다. 이는 데이터셋의 크기가 작아 과적합이 발생했을 가능성이 높다고 판단되었다. 이후, 파라미터 수가 적은 EfficientNet(B5) 모델로 실험했을 때, 검증 및 테스트 F1-score가 유사하게 나타나면서 데이터셋의 한계를 극복하는 데 적합한 모델의 중요성을 확인했다. 이에 따라 30M 파라미터 미만의 모델 중에서 작은 데이터셋에 적합한 모델을 찾기 위해 Tiny ViT를 선택했고, 이 모델이 테스트 F1-score에서 가장 높은 성능을 달성했다.

<br>

## Strategy 3. Multi-Task Learning (MTL)

<div align="center" >
 
![image](https://github.com/boostcampaitech6/level1-imageclassification-cv-09/assets/46400961/14b95703-4c2f-4a3a-bcc3-0a024709aaf3)

</div>

멀티 클래스 학습은 위에서 소개한 18개 클래스를 단일 태스크로 처리하는 반면, 멀티태스크 학습은 하드 파라미터 공유와 소프트 파라미터 공유, 두 가지 접근 방식을 통해 문제를 다룬다. 비교 결과, 멀티 클래스 학습과 하드 파라미터 공유를 사용한 멀티태스크 학습은 F1-score에서 비슷한 성능을 보였지만, 멀티 클래스 학습이 정확도에서 미세하게 더 우수했다. 그러나 소프트 파라미터 공유 방식을 적용한 멀티태스크 학습은 F1-score에서도 안정적인 성능을 유지하면서 정확도 면에서 가장 높은 결과를 달성했다.

<br>

## Strategy 4. Loss Function
Cross Entropy Loss를 사용했을 때 특정 클래스를 정확하게 분류하는 데 어려움을 겪었는데, 이는 클래스 간 불균형 문제에서 비롯된 것으로 분석되었다. Cross Entropy Loss는 이미 잘 분류하는 클래스에 더 초점을 맞추는 경향이 있어, 클래스 불균형 문제를 해결하기에는 적합하지 않다고 판단했다. 이에 따라, 비교적 쉽게 분류할 수 있는 클래스에는 패널티를, 분류하기 어려운 클래스에는 가중치를 부여하여 이 문제를 해결하기 위해 Focal Loss 함수를 도입했다. Focal Loss를 적용한 결과, 이전에 F1-score가 낮았던 클래스의 성능이 크게 향상되었으며, 전체적인 F1-score도 상승하였다.

<br>

## Strategy 5. Fine-tune

<div align="center" >
 
![image](https://github.com/boostcampaitech6/level1-imageclassification-cv-09/assets/46400961/e785eaeb-cf53-4984-a194-608778f67951)

</div>

Fine-tuning에는 세 가지 주요 방식이 있다. 1) 모델 전체를 학습시키는 방법, 2) 일부 레이어를 고정한 채 나머지 레이어를 학습시키는 방법, 그리고 3) 분류기를 제외한 모든 레이어를 고정하고 분류기만 학습시키는 방법이 존재한다. 본 대회는 데이터셋의 크기가 제한적이며 얼굴 이미지를 다루는 특성상, 두 번째 방식인 일부 레이어만 학습시키는 방법을 선택했다. 이 결정은 사전 학습된 모델의 초기 레이어가 일반적인 특징을 학습하는 반면에 후반부 레이어는 보다 복잡하고 특정 도메인에 특화된 고차원적 특징을 학습한다는 사실에 기반한다. 얼굴 데이터셋 같이 특정 분야에서 중요하고 세밀한 특징을 잘 포착하기 위해서는, 후반부 레이어의 세분화된 학습이 더욱 효과적이라고 판단했다. 이 방법을 적용함으로써, 기존 대비 F1-score가 5% 이상 개선되는 결과를 달성했다.

<br>
    
## Strategy 6. Model Ensemble
실험 과정에서 Top 1 모델이 훈련 데이터에 과적합되는 현상을 관찰했다. 이 문제를 해결하기 위해, 우리는 앙상블 기법을 도입했다. 구체적으로, 우리는 기존 데이터셋, 데이터를 제거한 데이터셋, 레이블을 수정한 데이터셋, 그리고 stage-2 데이터셋 등 총 네 가지 다양한 데이터셋을 사용했다. 이들 각각에 대해 Top 1 모델 아키텍처를 학습시킨 후, 이 네 개의 모델을 결합하여 소프트 보팅(Soft Voting) 앙상블 방식을 적용했다. 이 접근법을 통해 모델의 F1-score가 소폭 상승하는 결과를 달성했다.

<br>






