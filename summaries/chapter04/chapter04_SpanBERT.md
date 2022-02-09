​                       

# Chapter 4 BERT 파생 모델 2

이번 장에서 다루는 BERT의 다양한 형태의 파생 모델

- ALBERT
- RoBERTa
- ELECTRA
- SpanBERT

## SpanBERT

📄  **[SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/pdf/1907.10529.pdf)**

SpanBERT란 text span을 더 잘 표현하고 예측하기 위해 설계된 사전 학습 기법이다.

- 다음과 같은 방식으로 BERT를 확장하였다.

|                                               |                SpanBERT                |                  BERT                   |
| :-------------------------------------------: | :------------------------------------: | :-------------------------------------: |
|                **마스킹 방법**                |        연속된 랜덤 spans 마스킹        |            랜덤 토큰 마스킹             |
| **마스킹된 토큰 예측하기 위해 사용하는 표현** | span 경계 표현을 학습시켜서 예측 (SBO) | 마스킹된 개별 토큰의 표현을 이용해 예측 |

<div style="text-align:center">SpanBERT와 BERT</div>

🤔 기존 많은 NLP tasks는 **두 개 이상의 text span의 관계 추론**을 포함했고, 이는 self supervision tasks를 더 어렵게 했다.

➡︎ SpanBERT는 **span-level** 사전학습 방식으로 항상 BERT보다 좋은 성능을 낸다.

- 특히 <u>span selection tasks</u>에서 뛰어난 성능을 보인다.

  - question answering
- coreference resolution (상호참조해결)
- 데이터를 더 추가하거나 모델 크기를 키우지 않고, 좋은 사전학습 tasks와 objectives만으로도 좋은 성능을 낼 수 있다는 의의가 있다.

### Model

- 연속된 랜덤 spans 마스킹
- span boundary objective (SBO) 도입

- 단일 연속 text segment 샘플링  ➡︎ BERT의 NSP 생략

#### Span Masking

`masking spans of full words using a geometric distribution based masking scheme`

- 주어진 토큰 시퀀스 *X=(x1,…,xn)* 에서 masking budget(e.g. 15% of X)이 다 사용될 때까지 text span 샘플링

1. 각 iteration 마다 span length(단어 개수)를 샘플링

   - 짧은 span으로 치우친 geometric distribution *𝓵 ~ Geo(p)* 에서 샘플링

2. 마스킹될 span의 시작점을 랜덤(균일)하게 뽑는다.

   완전한 단어 seqeunce를 샘플링한다. (subword tokens X)

3. span에 있는 모든 토큰들을 [MASK] 또는 sampled tokens로 대체한다. (span-level masking)

   *Cf. BERT는 80-10-10% 로 각 토큰을 개별적으로 대체*

#### Span Boundary Objective (SBO)

`optimizing an auxiliary span boundary objective (SBO) in addition to MLM`

span selection model은 일반적으로 boundary 토큰을 이용하여 span의 고정된 길이 표현을 생성한다.

➡︎ **boundary 토큰의 표현만을 이용해 masked span의 각 토큰을 예측하는 <u>Span boundary objective(SBO)</u>** 도입했다.

- SBO는 모델이 fine-tuning 시에 쉽게 접근할 수 있는 boundary 토큰에 span-level 정보를 저장하게 하여 span selection model을 지원한다.

- span에 있는 마스킹된 토큰  `xi` *(target token: xi)* 을 예측하는 방법
  - xs와 xe를 각각 마스킹된 토큰에 대한 표현의 시작과 종료 지점이라고 할 때, 다음 3개의 값을 사용한다.
    - external boundary tokens `xs−1` and `xe+1`
    - position embedding of the target token `pi−s+1` (left boundary token xs-1로부터 상대 위치)

  - 3개의 값을 2 layer feed-forward network with GeLU activations and layer normalization 에 입력하여 얻은 출력값을 사용한다.

- SpanBERT의 손실함수는 MLM과 SBO loss *(cross-entropy loss)*를 더한 값이다.

#### Single-Sequence Training

BERT의 examples는 두 개의 text sequence *(X_A, X_B)* 를 사용하고, 두 sequence의 연결 여부를 예측*(NSP)*하게 모델을 학습시켰다. *(bi-sequence training with NSP)*

하지만 single-sequence training이 bi-sequence training with NSP보다 우수했고, 이유는 다음과 같을 것이라고 가정했다.

*: bi-sequence training은 모델이 더 긴 범위의 features를 학습하는 것을 방해하고, 결과적으로 많은 downstream tasks 에서 성능을 저하시킨다.*

➡︎ **NSP objective과 two-segment sampling procedure를 모두 제거하고,** 두 개의 half-segments를 합쳐서 n개의 토큰이 아닌, 

**최대 n=512의 토큰의 단일 연속 segment를 샘플링**해서 다양한 downstream tasks의 성능을 향상시켰다.

---

![An illustration of SpanBERT training. The span an American football game is masked. The SBO uses the output representations of the boundary tokens, x4 and x9 (in blue), to predict each token in the masked span. The equation shows the MLM and SBO loss terms for predicting the token, football (in pink), which as marked by the position embedding p3, is the third token from x4.](https://mitp.silverchair-cdn.com/mitp/content_public/journal/tacl/8/10.1162_tacl_a_00300/9/00300f01c.png?Expires=1647335036&Signature=2Z94mvu96t5Rlhgzz07oMo1qE11rmtqyU3ZlAozrJPX~h3m8vZa9siuGBi~Ni4uHuPCJjKD4E0n0WafeXohbpmcDboOJufNCXQIoHUU4FCsvx3nWQQmxVVFbSL1OOdHfrBAS-dggk3Fmoqgy9lxMWaOF-IfvhDN-M0FZDp7h6hqgOPjQOSGyo1rhxV9o8qcFAHvftZj7llwQ4SLmx1Tx7uJ~p5cAK2sZtzDfqNdqSQvc2~gM2eCPykLw72Wen1p-jRaqNlu6y76YBAfSbcphHD6mhPx0K8fnPdAbGA7GlIgpylRCay8lopV0p0-pFQ6JmOdpdvEyvWHlbVjekxgMUg__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)

<div style="text-align:center">SpanBERT 학습 과정</div> 

### Tasks

- Extractive Question Answering
  - 짧은 글과 질문을 입력한 후에 글에서 답으로 연속적인 text span을 선택하는 task

- Coreference Resolution

  - 동일한 개체(entity)를 표현하는 다양한 단어(mention)들을 찾아 연결해주는 task

    *[Coreference Resolution 관련 설명 및 논문이 정리된 블로그](https://jjdeeplearning.tistory.com/26)*

- Relation Extraction

  - 두 span이 들어있는 한 문장이 주어졌을 때 42개의 사전 정의된 관계 타입으로부터 span 사이의 관계를 예측하는 task

- GLUE

  - General Language Understanding Evaluation (GLUE) benchmark는 9개의 sentence-level 분류 tasks로 구성

### Results

- SpanBERT는 거의 모든 task에 대해 BERT보다 뛰어나다.
- SpanBERT는 특히 extractive question answering에 좋다.
- single-sequence training이 bi-sequence with NSP 보다 상당히 잘 작동한다.

### Summary

- 이 논문은 span 기반 사전 학습 방식인 SpanBERT를 제시했다.

- SpanBERT는 BERT를 확장했다.

  ​	(1) random 토큰 대신, 연속적인 랜덤 span을 마스킹한다.

  ​	(2) 개별 토큰 표현이 아닌, span 경계 표현을 학습시켜 마스킹된 span의 전체 내용을 예측한다.

- SpanBERT는 여러 task에 대해 모든 BERT baselines를 뛰어넘었다.

- SpanBERT는 span selection tasks에 특히 좋은 성능을 보인다.

## 사전 학습된 SpanBERT를 질문-응답 태스크에 적용하기

```python
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/spanbert-large-finetuned-squadv2",
    tokenizer="SpanBERT/spanbert-large-cased"
)

results = qa_pipeline({
    'question': "What is machine learning?",
    'context': "Machine learning is a subset of artificial intelligence. It is widely for creating a variety of applications such as email filtering and computer vision"
})

print(results['answer'])  # a subset of artificial intelligence
```

*SpanBert는 텍스트 범위를 예측하는 작업에 많이 사용된다.*

## 참고 자료

- 구글 BERT의 정석 [book](http://www.yes24.com/Product/Goods/104491152)