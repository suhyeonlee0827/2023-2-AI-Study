# 0920 강의

# Computer Vision Application

what is Computer Vision?

A field of AI that enables computers and systems to **derive meaningful information from digital images, videos and other visual inputs** and take actions or make recommendations based on that information. 

**AI** enables computers to **think, computer vision** enables them to **see, observe and understand.**

적용 분야 ex) 자율주행

### 4 tasks of Computer Vision

1. Classification
2. Object Detection
3. Image Segmentation
4. Visual Relationship

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/824bec17-b72f-45d3-acf7-41d75aa4882e/f64eaf0d-bd55-4854-ba90-47442a727603/Untitled.png)

Segmentation: a key technique in machine learning that allows you to divide data into groups with similar characteristics.

입력 이미지를 픽셀 수준에서 분석해 각 픽셀에 클래스 레이블을 할당.

- Semantic Segmentation: a deep learning algorithm that associates a label or category with every pixel in an image.
- Instace Segmentation: detecting instances of objects and **demarcating** their boundaries.
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/824bec17-b72f-45d3-acf7-41d75aa4882e/3a7f94fb-bd31-4a46-b028-a608657ff05a/Untitled.png)
    
- image 내에 2명의 사람, 3개의 차가 있다고 한다. 사람이라는 동일한 class  내에서 사람A, 사람B를 나타내는 것을 instance라고 한다.
    
    A,B,C는 모르겠지만 일단 사람과 차의 픽셀 위치를 찾아내는 것을 Semantic segmentation이라고 한다.
    
    사람 A, 사람 B, 차 A, 차 B, 차 C 이렇게 분류해 내는 것을 Instance segmentation 이라고 말한다.
    

# 2. Semantic Segmentation

### FCN vs CNN 차이

CNN : Convolution Neural Network

- **Fully Connected Layer** (=Dense layer)
- 뒷 부분의 layer로 갈 수록 ****************spatial information****************이 ********************************************사라진다. FCL에서는 거의 모두 없앤다.
- 왜 사라지는가?
    
    **down sampling**을 통해 입력 이미지의 W*H는 점점 작아지게 된다. 자연스럽게 이미지 내의 공간 정보도 사라진다. 
    

구조

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/824bec17-b72f-45d3-acf7-41d75aa4882e/0304c91d-96c9-429f-b121-732a73dcb97c/Untitled.png)

FCN : Fully Convoultional Network

- filter들이 모두 conv filter (=fully convolution network)
- Convolution layer를 1*1 필터로 적용해 대체한다. Output의 형태가 공간적 정보를 보존해 upsampling 시 input과 같은 크기의 출력을 생성 가능.
- 공간적 정보를 무시한 벡터로 표시
- learn representations and make decisions based on local spatial input.
- FC layer가 제거 되었기 때문에 input image size가 고정될 필요가 없다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/824bec17-b72f-45d3-acf7-41d75aa4882e/d00320b5-fa53-40d2-9a0e-9b853bf5e681/Untitled.png)

### FCL vs CL

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/824bec17-b72f-45d3-acf7-41d75aa4882e/c4696f32-e0c7-4395-a0e2-302c98c3935a/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/824bec17-b72f-45d3-acf7-41d75aa4882e/bd6a25a9-a6bb-4056-a2a4-38b87b67e968/Untitled.png)

- upsampling
    - deconvolution
    - interpolation(보강법)

# 3. Object Detection

= bounding box로 객체의 위치를 찾는 task (classification & localization)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/824bec17-b72f-45d3-acf7-41d75aa4882e/1b8c1e02-0107-4cf8-9856-c3d2d41eee77/Untitled.png)

- AlexNet: a type of CNN architecture
    - 기존 영상처리 알고리즘들의 50%대의 성능을 80%로 끌어올림.

### R-CNN

two - staged detector (순차적으로 2 task 진행, classification & localization) 

1> **Region Proposal**

- 한 image 내에 object로 추정되는 2000개의 후보를 CNN(ex. Alexnet)에 전달
    
    ![Example of the Selective Search algorithm](https://prod-files-secure.s3.us-west-2.amazonaws.com/824bec17-b72f-45d3-acf7-41d75aa4882e/89efcf40-33c2-47f5-851e-7624da9e7161/Untitled.png)
    
    Example of the Selective Search algorithm
    

2> **CNN computation**

- 전 단계에서 제안된 region을 동일한 크기로 resize 작업을 선행합니다. (warped region)
- 이후 기존 CNN처럼 연산해서 feature vector 생성

3> **Classification**

- classification : object 여부와 class (by Linear SVM)
- 자세히
    
    1) Region Proposal
    
    selective search(merging and splitting segments of image based on image cues like color, texture, and shape), edgeboxes 등의 알고리즘을 통해 region proposal(후보영역)을 추출해 warp.
    
    이 후보영역을 ROI(Region of Interest) 라고 한다.
    
    2) Feature Extraction
    
    - resize region (warp region) : CNN은 input size가 동일해야 하므로 resize (227X227)
    - CNN으로 feature vector까지 연산
    
    3) Object Classification 
    
    - use pretrained CNN model. fine tuning
    - feature vector를 SVM을 이용해서 각 class에 대한 score를 계산
    - score가 부여된 region들을 아래 기준에 의해 선별한다.
        - IoU가 threshold(0.5)를 넘어야 한다.
            - **Positive sample** : IoU가 0.5 이상인 region
            - **Netgative sample** : Positive sample이 아닌 region (background)
        - **NMS(non-maximum suppression)** : 겹치는 region들 중에서 가장 confidence가 높은 것만 남기는 작업
    
    4) Bounding Box Regression
    
    R-CNN vs R-CNN BB. BB regression을 사용했을 때 성능이 향상된다.
    

단점:

- computational complexity
- slow inference: 2000개의 region 추출 및 연산. 병목현상도 발생.
- overlapping region proposals
- not end-to-end

### SPPNet

- fc layer 이전에 spatial pyramid pooling layer를 추가해 convolutional layer가 임의의 사이즈로 입력을 취할 수 있게 함.
    
    ⇒ 입력 이미지를 crop/warp 할 필요성을 없앰. 
    
    ⇒ 입력 이미지의 scale, size, aspect ratio에 영향 X.
    
- 2000번의 CNN 과정 → 1번의 CNN(convolution 한 번만)
- 2000개의 영역에 해당하는 피처값인 서브텐서만 가져와 속도 향상

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/824bec17-b72f-45d3-acf7-41d75aa4882e/f39d8f7e-e0be-4ce4-affc-70f798d2a10e/Untitled.png)

### Fast R-CNN

- ROI Pooling: 1개의 피라미드 SPP로 고정된 크기의 feature vector를 만드는 과
- End-to End
    
    R-CNN에서는 CNN을 통과한 후 각각 서로 다른 모델 SVM(classification), bounding box regression(localization) 안으로 들어가 forward됐기 때문에 연산이 공유되지 않았다.
    
    (* bounding box regression은 CNN을 거치기 전의 region proposal 데이터가 input으로 들어가고 SVM은 CNN을 거친 후의 feature map이 input으로 들어가기에 연산이 겹치지 않는다.)
    
    이제 RoI영역을 CNN을 거친 후의 feature map에 투영시킬 수 있었다.
    
    따라서 동일 data가 각자 softmax, bbox regressor으로 들어가기 전에 연산을 공유한다.
    
- nerual network을 통해 bbox regressor & softmax 수행

단점 : Selective Search는 CNN 외부에서 진행되므로 아직도 속도의 bottleneck

### Faster R-CNN

- Region Proposal Network 도입. detection에서 쓰인 conv feature를 RPN에서도 공유해 ROI생성 역시 CNN level에서 수행해 속도 향상. selective search를 사용하지 않는다.
- bbox region을 더 정확하게 계산하도록 학습가능한 Network.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/824bec17-b72f-45d3-acf7-41d75aa4882e/445b9fae-7705-40c7-8fec-437f4e833d93/Untitled.png)

### Yolo: You Only Look Once

- RPN 단계X

1) 가로 세로를 동일한 그리드 영역으로 나누기

2) 각 그리드 영역에 대해서 어디에 사물이 존재하는지 BB와 박스에 대한 신뢰도 점수를 예측(신뢰도가 높을수록 굵게 박스를 그린다. ) & 어떤 사물인지에 대한 classification 작업을 동시에 진행

3) 굵은 박스들만 남긴다. (NMS 알고리즘)

# 4. Intro to Pytorch

딥러닝 프레임워크 : Tensorflow, Pytorch, keras, Jax

Jax: 구글에서 개발하고 유지관리하며 사용. 구글 연구진이 한계를 뛰어넘기 위해 사용함. 공식적으로 구글 제품 X. 일부 코드 실행 속도가 4~5배 높다. 어려워서 대중화되진 않았음.

- tensorflow vs pytorch
    - tensorflow: 배포, 운영, 라이브러리 풍부하게 지원.
        - **static graph**- 모든 연산이 미리 정의, 그 연산들의 그래프가 구축되어있음.
        - 지금은 버전 업그레이드를 통해 동적 그래프도 가능.
    - pytorch: 배포에 부진.
        - **dynamic graph** - 연산을 실행하는 동안 그래프가 생성되고 수정 가능.

## 5. Pytorch Basics

- tensor
    - 다차원 Arrays 개념. numpy의 ndarray와 비슷.
    - list나 ndarray를 사용해 생성 가능.
- operations (numpy like)
    - slicing, indexing, flatten, ones_like, numpy, shape, dtype
    - tensor handling
        - view - memory 주소 copy (contiguous) [shallow copy]
        - reshape - value copy [deep copy]
        - unsqueeze
        - matmul 은 broadcasting 지원 O.
            
            mm은 지원 X
            
        - nn.functional
    - AutoGrad
        - backward
    

## 6. Pytorch 프로젝트 구조 이해하기

## 7. AutoGrad & Optimizer

- @ 뜻은 행렬곱

## 8. Pytorch Dataset

## 9. Model Save

- 딥러닝 모델은 평균적으로 학습시간이 굉장히 길다. 12시간 정도.

## 10. Transfer Learning

- pretrained model 사용하자
- transfer learning: pretrained model로 내 데이터를 다시 학습시키는 것.

참고 자료
