# Chapter 8 sentence-BERT 및 domain-BERT 살펴보기

## 내용

- sentence-BERT
- 지식 증류로 다국어 임베딩 학습
- domain-BERT (ClinicalBERT 및 BioBERT)

## 지식 증류를 이용한 다국어 임베딩 학습

*Q. 영어 외의 다른 언어에는 어떻게 sentence-BERT를 사용할까?*

*A. sentence-BERT에서 생성된 단일 언어 문장 임베딩을 **지식 증류**를 통해 다국어로 만들어 다양한 언어에 sentence-BERT를 적용할 수 있다.*

sentence-BERT 지식 ➡︎ 다국어 모델 (e.g. XLM-R) ➡︎ 다국어 모델이 사전 학습된 sentence-BERT와 동일한 임베딩 형성

학생 모델 S로 계산한 소스 문장(s_j) 표현과 타깃 문장(t_j) 표현 모두 교사 모델 T로 계산한 소스 문장(s_j) 표현과 동일해지는 방향으로 다음 속성을 학습

1. 벡터 공간이 언어 간 정렬 i.e., 다른 언어의 같은 문장은 가까이 위치
2. 교사 모델 T로 계산한 소스 언어의 벡터 공간 특성을 채택하여 다른 언어로 전이



![Figure 1](https://d3i71xaburhd42.cloudfront.net/b63075f3249e0c7f7e92da49fc87fc7f9df48d4b/2-Figure1-1.png)  <img src="https://miro.medium.com/max/1400/0*LhYkSB4sfilxSykr" alt="knowledge distilation objective" style="zoom: 33%;" />

: 미니배치 B에 대한 평균 제곱 오차(MSE)들을 최소화하도록 학생 네트워크 학습

## domain-BERT

*BERT : 일반 위키피디아 말뭉치를 사용해 BERT를 사전 학습시키고 이를 파인튜닝해 다운스트림 태스크에 사용*

일반 위키피디아 말뭉치에서 사전학습된 BERT 사용 vs 특정 도메인 말뭉치에서 BERT를 처음부터 학습

➡︎ BERT가 특정(일반 위키피디아 말뭉치에 없을 수도 있는) 도메인 임베딩을 학습시키는데 도움

### ClinicalBERT

📄 [ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission](https://arxiv.org/pdf/1904.05342.pdf)

임상 텍스트의 콘텍스트 표현을 이해하기 위해 대규모 임상 말뭉치에서 사전학습된 임상 domain-BERT 모델

<img src="https://d3i71xaburhd42.cloudfront.net/b3c2c9f53ab130f3eb76eaaab3afa481c5a405eb/2-Figure1-1.png" alt="figure1" />

#### 사전 학습

- Medical Information Mart for Intensive Care III (MIMIC-III) dataset 사용

- BERT와 마찬가지로 MLM과 NSP 태스크를 이용해 사전 학습

#### 파인 튜닝

재입원 예측, 입원 기간, 사망 위험 추정, 진단 예측 등 다양한 다운스트림 태스크에 맞춰 파인 튜닝

![figure3](https://d3i71xaburhd42.cloudfront.net/b3c2c9f53ab130f3eb76eaaab3afa481c5a405eb/5-Figure3-1.png) 

<img src="/Users/seohui/Documents/진로/스터디/구글 BERT의 정석 /Getting-Started-with-Google-BERT/images/chap8_probability-of-readmission1.png" alt="image-20220316184710938" style="zoom:50%;" /> 

*e.g. 30일 이내 재입원 예측 테스크*

- 사전학습된 ClincalBERT에 임상 메모를 입력하고 임상 메모의 표현 반환
- `[CLS]` 토큰의 표현을 가져와 분류기(피드포워드 + 시그모이드 활성화 함수)에 입력
- 분류기는 30일 이내에 환자가 다시 입원할 확률 반환

#### Empirical Study

**Empirical Study I: Language Modeling and Clinical Word Similarity**

- Clinical language model에서 ClinicalBERT가 BERT보다 좋은 성능

<img src="https://d3i71xaburhd42.cloudfront.net/b3c2c9f53ab130f3eb76eaaab3afa481c5a405eb/6-Table1-1.png" alt="table1" />

- 임상 단어 유사도 추출

  - ClinicalBERT에서 배운 표현을 경험적으로 평가하기 위해 의학 용어 표현 계산 

    <img src="https://d3i71xaburhd42.cloudfront.net/b3c2c9f53ab130f3eb76eaaab3afa481c5a405eb/7-Figure4-1.png" alt="figure4" style="zoom:50%;" />

  - 신체기관이 서로 관련된 의학 용어끼리 가까이 있는 것을 확인 가능 

    ➡︎ ClinicalBERT의 표현이 의학 용어에 대한 콘텍스트 정보를 갖고 있음을 나타냄

**Empirical Study II: 30-Day Hospital Readmission Prediction**

- Scalable Readmission Prediction

  **💡 BERT 최대 토큰의 길이 512보다 더 많은 토큰으로 구성된 경우?**

  - 512보다 더 많은 토큰으로 구성된 긴 시퀀스를 여러 서브시퀀스로 분할
  - 각 서브시퀀스를 모델에 입력한 후 모든 서브시퀀스를 개별적으로 예측
  - 다음 식을 이용해 점수 계산 : <img src="/Users/seohui/Documents/진로/스터디/구글 BERT의 정석 /Getting-Started-with-Google-BERT/images/chap8_probability-of-readmission2.png" alt="image-20220316183101249" style="zoom:50%;" />
    - 재입원 예측과 관련된 서브시퀀스 : 확률이 높은 것
    - 노이즈가 포함된 서브시퀀스 방지를 위해 평균 확률 사용
    - 평균 확률에 중요성 부여

### BioBERT

📄 [BioBERT: a pre-trained biomedical language representation model for biomedical text mining](https://arxiv.org/pdf/1901.08746.pdf)

생물 의학 텍스트 이해를 위해 대규모 생물 의학 코퍼스에서 사전학습된 생물 의학 domain-BERT

![Figure 1](https://d3i71xaburhd42.cloudfront.net/1e43c7084bdcb6b3102afaf301cce10faead2702/2-Figure1-1.png)

#### 사전 학습

- 생물 의학 도메인 텍스트를 이용해 사전학습

  - PubMed, PMC

- 사전학습 전에 먼저, 영어 위키피디아 및 토론토 책 말뭉치 데이터셋으로 구성된 일반 도메인 말뭉치를 사용해 사전 학습된 일반 BERT로 BioBERT의 가중치 초기화

- 워드피스 토크나이저로 토큰화

  - 생물 의학 코퍼스의 새로운 어휘를 사용하는 대신, BERT 기반 모델에서 사용되는 원래 어휘로 워드피스 어휘를 구성

    - BioBERT와 BERT 간의 호환성

    - 본 적 없는 단어도 원래 BERT 기반 어휘를 사용해 표현하고 파인튜닝

- 대소문자가 있는 어휘를 사용하는 것이 다운스트림 태스크에서 더 좋은 성능을 얻을 수 있음을 발견

#### 파인 튜닝

📌 코드 [github.com/dmis-lab/biobert](https://github.com/dmis-lab/biobert?fbclid=IwAR3te48b9-SsBBmybH8MrQMLxX5RlDbYTfpBZQGLp5ki1B8jq-2M15t3skA)

**개체명 인식(NER) 태스크를 위한 BioBERT**

- 생물 의학 코퍼스에서 특정 도메인의 다양한 고유명사 인식
- BERT를 파인튜닝하는 법과 동일
- 데이터셋
  - 질병 관련된 개체명: NCBI, 2010 i2b2/VA, BC5CDR
  - 약물 및 화학 관련된 개체명: BC5CDR, BC4CHEMD
  - 유전자와 관련된 개체명: BC2GM, JNLPBA
  - 종과 관련된 개체명: LINNAEUS, Species-800

**관계 추출(RE)을 위한 BioBERT** 

- 생물 의학 코퍼스에서 명명된 개체들의 관계 분류

- `[CLS]` 토큰 표현을 사용하는 BERT의 sentence classifier 사용

**질문-응답(QA)을 위한 BioBERT**

- BioASQ 데이터셋으로 파인튜닝
  - 생물 의학 질문-응답 데이터셋으로 널리 사용
  - SQuAD의 형식과 동일
  - BERT를 파인튜닝하는 법과 동일

