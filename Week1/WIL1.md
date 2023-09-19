0913 강의 내용 정리
1. AI >> ML(machine learning) >> DL(deep learning) >> Neural network 
2. Deep learning components
    1. data
    2. Model: input에서 feature를 뽑아 원하는 output을 만들어내는 프로그램
    3. loss function: 얼마나 잘못 예측한 지에 대한 지표
        - 다루는 task에 dependent
    4. optimization (최적화 기법) and regularization (일부러 학습 방해)
3. Neural Network
4. Nonlinear functions
5. Multi-Layer Perceptron (MLP 구조)
6. generalization
    1. under-fitting vs over-fitting vs optimal balance
    2. cross-validation (test data를 학습시키는 것은 cheating, 따라서 valid data 필요)
    3. ensemble: 여러 개의 분류 모델을 모으고 결합시켜 더 정확한 예측을 하는 것.
        
        ex) bagging & boosting
        
    4. regularization
        1. early stopping
        2. parameter norm penalty
        3. data augmentation
        4. noise robustness - 이상치나 노이즈가 들어와도 크게 흔들리지 않음.
        5. Dropout
        6. label smoothing

6. 합성곱 신경망 (Convolutional Neural Networks, CNN)

- fnn vs cnn
    - 인접 픽셀 간의 상관관계가 무시되어 이미지를 벡터화하는 과정에서 정보손실 발생
- convolution 계산법
- RGB 계산법
- pooling 계산법
1. 1*1 convolution
    - depth 차원 변경 가능 → neural network 깊게 쌓기 가능
2. Modern CNN
    - 유명 대회 : ILSVRC
        - Classification, Detecton, Localization, Segmentation 을 다룸
        - AlexNet (2012 winner)
            - 2개의 네트워크
            - 이전 모델들과 비교 : 비선형 함수 사용/ data augmentation / dropout
        - VGGNet (2014  준 winner) - 3*3 convolution
        - GoogLeNet (2014 winner) - 1*1 convolution
        - ResNet (2015 winner) - 사람의 능력을 뛰어넘은 첫 모델. skip connection 사용
        
    
