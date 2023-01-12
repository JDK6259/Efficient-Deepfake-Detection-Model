# Efficient-Deepfake-Detection-Model
> 효율성을 향상시킨 딥페이크 탐지 모델 

## Description
1. Pytorch Retinaface
   |___ video2crop.py : 3종류의 딥페이크 데이터셋을 전처리함. 비디오에서 얼굴 부분을 crop하여 이미지로 저장함.

2. SimCLR
   |___ data_aug
           |___ make_label_df.py : 데이터 별 학습에 활용할 이미지 ID 및 경로 등을 정리하여 csv 파일로 저장함.
           |___ contrastive_learning_dataset.py : 학습에 활용할 이미지를 불러오는 DeepFakeDataset class를 정의하고,
				     SimCLR 학습방식에 맞는 data transform을 적용하기 위한 ContrastiveLearningDataset class 또한 정의함.
   |___ models
           |___ resnet_simclr.py : resnet의 말단 부분을 수정하여 SimCLR 학습 방식에 맞는 구조로 backbone 수정함.
   |___ simclr.py : 논문 상의 SimCLR을 구현함.
   |___ runs : SimCLR 학습 log 및 최종 모델 가중치 저장
           |___ 014 : RGB channel + resnet50
           |___ 034 : gray channel + resnet50
           |___ 036 : RGB channel + resnet18
           |___ 041 : gray channel + resnet18
           
 3. CNN_ConvLSTM.ipynb
: SimCLR 방식으로 학습된 CNN모델을 freeze 한 뒤 ConvLSTM과 연결하여 각 동영상에 대해 딥페이크 여부를 판단하는 학습 진행함.
 3가지 유형의 데이터셋에 대해 각각 모델 학습 진행함.


## References
- https://github.com/biubug6/Pytorch_Retinaface
- https://github.com/sthalles/SimCLR
