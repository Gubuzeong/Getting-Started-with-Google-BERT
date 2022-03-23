# Chapter 9. VideoBERT

## VideoBert로 언어 및 비디오 표현 학습

**VideoBERT** 📄 [VideoBERT: A Joint Model for Video and Language Representation Learning](https://arxiv.org/pdf/1904.01766.pdf)

![text-to-video generation and future forcasting examples](https://d3i71xaburhd42.cloudfront.net/c41a11c0e9b8b92b4faaf97749841170b760760a/2-Figure2-1.png)

- 영상과 언어의 표현을 동시에 배우는 최초의 BERT 모델
- 이미지 캡션 생성, 비디오 캡션, 비디오의 다음 프레임 예측 등과 같은 태스크에 사용

### VideoBERT 사전 학습

- **MLM(cloze 태스크)과 언어-시각(linguistic-visual) 정렬이라는 새로운 태스크를 사용하여 사전학습**

- 데이터: 교육용 비디오(e.g., 요리 비디오)

  - 교육자의 말과 해당 시각 자료(영상)가 서로 일치해 언어와 비디오의 표현을 동시에 배우는데 도움

- 비디오에서 언어 토큰과 시각 토큰을 추출하여 학습에 사용

  - 언어 토큰 추출 방법

    1. 비디오에 사용된 오디오를 추출

    2. 오디오를 텍스트로 변환

       **자동 음성 인식**(automatic speech recognition) 툴킷을 활용
       _Cf. https://cloud.google.com/speech-to-text_

    3. 텍스트를 토큰화하면 언어 토큰 생성

  - 시각 토큰 추출 방법

    1. 비디오의 이미지 프레임을 20fps(초당 프레임)로 샘플링
    2. 이미지 프레임을 1.5초의 구간으로 시각 토큰들로 변환

- 입력 토큰

  1. 언어와 시각 토큰 결합

     특수 토큰 `[>]` 으로 언어 및 시각적 토큰 결합 

  2. 언어 토큰의 시작 부분에 `[CLS]` 토큰 추가, 시각 토큰 끝에만 `[SEP]` 토큰 추가

  3. 언어 및 시각 토큰 중 몇 가지 토큰들을 무작위로 마스킹

- **cloze 태스크**

  - VideoBERT에서 반환된 마스크된 표현을 분류기(피드포워드 + 소프트맥스)에 입력하면, 분류기는 <u>마스크된 토큰 예측</u>

  ![cloze task](https://d3i71xaburhd42.cloudfront.net/c41a11c0e9b8b92b4faaf97749841170b760760a/4-Figure3-1.png)

- **언어-시각 정렬**

  - NSP 태스크와 유사한 분류 태스크

  - 언어와 시각 토큰이 시간적으로 서로 정렬되어 있는지 예측

    = <u>텍스트(언어 토큰)가 비디오(시각적 토큰)와 일치하는지 여부 예측</u>

  - `[CLS]` 토큰의 표현을 가져온 다음 주어진 언어 및 시각 토큰이 일시적으로 정렬되는지 여부를 분류하는 분류기에 입력

    _Cf. NSP는 `[CLS]` 표현을 이용해 주어진 문장 다음에 출현하는 문장이 다음 문장인지 예측_

- **최종 사전 학습 목표**

  - 세 가지 목표: 텍스트, 비디오, 비디오-텍스트

    - **텍스트**

      언어 토큰을 마스킹하고 마스크 언어 토큰을 예측하도록 모델을 학습시켜서 모델이 언어 표현을 더 잘 이해하도록 함

    - **비디오**

      시각 토큰을 마스킹하고 마스크된 시각 토큰을 예측하도록 모델을 학습시켜서 모델이 비디오 표현을 더 잘 이해하도록 함

    - **비디오-텍스트**

      언어 및 시각적 토큰을 마스킹하고 모델을 학습시켜 언어 및 시각 토큰을 예측하고, 언어-시각 정렬을 학습하게 해서 모델이 언어와 시각 토큰 간의 관계를 이해하게 함

  - 최종 사전 학습 목표: 세 가지 방법을 모두 활용한 가중치 조합
  - 4개의 TPU를 사용해 2일 동안 50만 번 반복해 최종 사전 학습 목표를 수행하며 학습

### 데이터 소스 및 전처리

**데이터셋**

- 유튜브의 교육용 비디오 사용

  - 유튜브 동영상 주석(annotation) 시스템을 이용해 요리와 관련된 유튜브 동영상 추출

    - 15분 미만 영상 길이, 31만 2000개의 동영상

  - 유튜브 API에서 제공하는 자동 음성 인식 도구(ASR)를 사용해 텍스트 추출

    - 텍스트를 타임 스탬프와 함께 갖고오기 위함 (비디오에 사용된 언어에 대한 정보도 반환)

    - 31만 2000개의 동영상 중 18만 개의 동영상에만 ASR을 적용할 수 있었고, 그 중 영어로 된 동영상은 12만개로 추정

      ➡︎ 텍스트 및 비디오-텍스트 목표를 위해 12만 개의 비디오만 사용하고, 비디오 목표는 31만 2000개의 비디오 사용

**비디오 및 언어 전처리**

- 시각 토큰
  1. 비디오의 이미지 프레임을 20fps(초당 프레임)로 샘플링
  2. 이미지 프레임을 1.5초의 구간으로 시각 토큰들로 변환 (= 30-frame clip 생성)
  3. 각 30-frame clip에 사전 학습된 비디오 컨볼루셔널 뉴럴넷을 적용해 특징 추출
  3. 계층적 k-평균 알고리즘을 적용해 시각 특징 토큰화
- 언어 토큰
  1. 상용 LSTM 기반 언어 모델을 이용해 각 ASR 단어 시퀀스에 구두점을 추가하여 단어 스트림을 문장으로 나눔
  2. 각 문장에 대해 BERT의 텍스트 전처리와 동일한 방식을 따르며, 텍스트 토큰화 

> 💡 자연스럽게 문장으로 나뉘는 언어와 달리, 비디오는 어떻게 의미론적으로 나눌까?
>
> - 휴리스틱 사용
>   - ASR 문장이 존재할 경우, 문장의 시작 및 종료 timestamp 사이에 해당하는 비디오 토큰을 세그먼트로 취급
>   - ASR 문장이 존재하지 않을 경우, 하나의 세그먼트를 16개의 토큰으로 취급

### VideoBERT의 응용

사전학습된 VideoBERT 모델을 사용해 다양한 <u>다운스트림 태스크</u>에 맞춰 파인튜닝

- 시각 토큰을 입력해 상위 3개의 다음 시각 토큰 예측
- 텍스트가 주어지면 해당하는 비디오 생성
- 비디오에 자막 생성

## Transformers in vision

📄 [Transformers in Vision: A Survey](https://arxiv.org/pdf/2101.01169.pdf)

- Computer Vision 분야에서 Transformer가 활용된 연구들 정리한 survey paper

### Background

RNN to Transformer in NLP ⇨ CNN에 self-attention 적용 ⇨ Transformer 모델 자체를 CV 태스크에 사용

### Task

![TABLE 1: A summary of key design choices adopted in different variants of transformers for a representative set of computer vision applications.](https://d3i71xaburhd42.cloudfront.net/3a906b77fa218adc171fecb28bb81c24c14dcc7b/21-Table1-1.png)

***Table 1 from Transformers in Vision: A survey***

*A summary of key design choices adopted in different variants of transformers for a representative set of computer vision applications. The main changes relate to specific loss function choices, architectural modifications, different position embeddings and variations in input data modalities.*

### Model

#### Transformers for Multi-Modal Tasks 

*Multi-modal learning: 다양한 데이터 타입, 데이터 형태, 다양한 특성을 갖는 데이터를 사용하는 학습법*

- Transformer 모델은 <u>vision-language 태스크</u>에도 광범위하게 사용
  - visual question answering (VQA)
  - visual commonsense reasoning (VCR)
  - cross-modal retrieval
  - image captioning

![Fig. 12: An overview of Transformer models used for multi-modal tasks in computer vision](https://d3i71xaburhd42.cloudfront.net/3a906b77fa218adc171fecb28bb81c24c14dcc7b/14-Figure12-1.png)

*An overview of Transformer models used for multi-modal tasks in computer vision*

#### Vision Transformer (ViT) 

📄 [AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929.pdf)

![Fig. 6: An overview of Vision Transformer (on the left) and the details of Transformer encoder (on the right)](https://production-media.paperswithcode.com/models/Screen_Shot_2021-02-14_at_2.26.57_PM_WBwCIco.png)

- CNN 구조였던 computer vision 문제를 Transformer 구조로 대체
  - Transformer 구조를 사용한 Architecture가 수 많은 SOTA를 찍고 있으며, **ViT 논문이 그 시작점**
  - Transformer 구조를 활용하여 image classification을 수행한 방법론
  - 더 많은 데이터를 더 적은 비용으로 사전 학습

- 구조

  - Transformer encoder 사용
  - 한 이미지를 여러 patch로 분할 (patch를 단어같이 취급)
  - patch, classification token, position embedding을 입력하여 최종 classification 결과 생성

  > CNN vs Transformer
  >
  > - Layer
  >
  >   - CNN: 이미지 전체의 정보를 통합하기 위해서는 몇 개의 layer 통과
  >   - Transformer: 하나의 layer로 전체 이미지 정보 통합 가능
  >
  > - Inductive bias
  >
  >   _새로운 데이터에 대해 좋은 성능을 내기 위해 모델에 사전적으로 주어지는 가정_
  >
  >   - CNN: 2차원의 지역적인 특성 유지, 학습 후 weight 고정
  >
  >     → 인접한 픽셀 간 강한 상관관계가 있다는 특징을 살려 inductive bias가 적절하게 만들어짐으로써 이미지 특징 효과적 추출
  >
  >   - Transformer: 1차원 벡터로 만든 후 self attention (2차원의 지역적인 정보 유지 x), weight가 input에 따라 유동적으로 변함
  >
  >     → inductive bias가 적고, 모델의 자유도가 높아 데이터로부터 더 많은 정보를 얻을 수 있음

- 한계

  - 학습 데이터가 충분하지 않을 경우 CNN 모델보다 성능 감소 (∵ inductive bias ↓)

    ➡︎ 대용량의 학습 자원과 데이터가 필요

#### Data efficient image Transformer (DeiT)

📄 [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877)

- 많은 데이터가 필요한 ViT 한계 극복
  - Knowledge Distilation
  - Data Augmentation

## 참고 자료

#### 개념

- [Multimodal Deep Learning](https://towardsdatascience.com/multimodal-deep-learning-ce7d1d994f4)

#### 세미나

- [DMQA Transformer in Computer Vision](http://dmqm.korea.ac.kr/activity/seminar/316)

#### 논문

- [VideoBERT: A Joint Model for Video and Language Representation Learning](https://arxiv.org/pdf/1904.01766.pdf)
- [Transformers in Vision: A Survey](https://arxiv.org/pdf/2101.01169.pdf)
  - **리뷰**
    - [Transformers in Vision： A Survey [1] Transformer 소개 & Transformers for Image Recognition](https://hoya012.github.io/blog/Vision-Transformer-1/)
- [AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929.pdf)
  - **리뷰**
    - https://engineer-mole.tistory.com/133
    - https://kmhana.tistory.com/27

#### 책

- [구글 BERT의 정석](http://www.yes24.com/Product/Goods/104491152)
