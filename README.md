# Project
* 특정한 의상 item과 유사한 item 다른 item을 추천한다.

# Dataset
* polyvore dataset을 활용
> * 11만여개의 의상 item을 갖고 있음

# Method
1. 각 의상 item을 Resnet50을 통해 2,048차원의 벡터로 변환
> * user가 등록한 의상 item의 feature vector를 추출한다.
> * polyvore dataset에 포함된 모든 의상 item들의 feature vector를 추출하고 저장
2. cosine 유사도 함수를 활용하여 유사도 추정

# Future Work
* 각 의상 item의 category를 고려해야함
> * 현재 Network인 resnet50에는 category에 대한 정보가 없음
> * 향 후 condition으로 category에 대한 정보를 줘서 같은 category를 갖는 item을 보여줘야함
