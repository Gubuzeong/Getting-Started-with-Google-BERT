# Chapter 3. BERT 활용하기

## 사전 학습된 BERT 모델

*BERT를 처음부터 사전 학습시키는 것은 계산 비용이 많이 든다. 따라서 사전 학습된 공개 BERT 모델을 다운로드해 사용하는 것이 효과적이다.*
*Cf. [구글 리서치의 깃허브 저장소](https://github.com/google-research/bert) 또는 [사용 가능한 모든 사전 학습된 BERT 모델을 확인할 수 있는 huggingface.co](https://huggingface.co/models?sort=downloads&search=bert)*

### 종류

- 구조
  - 인코더 레이어 수 *L*, 어텐션 헤드 *A*, 은닉 유닛 *H*의 다양한 조합
    - BERT-base / BERT-large
    - BERT-tiny / BERT-mini / BERT-small / BERT-medium

- 형식

  - BERT-uncased : 모든 토큰이 소문자인 상태로 학습을 진행한 모델
    - 일반적으로 사용되는 모델

  - BERT-cased : 토큰에 대해 소문자화를 하지 않은 상태로 학습을 진행한 모델
    - 대소문자를 보존해야 하는 **개체명 인식(NER)**과 같은 특정 작업을 수행하는 경우


### 두 가지 사용 방법

- 임베딩을 추출해 특징 추출기로 사용한다.
- 사전 학습된 BERT 모델을 텍스트 분류, 질문-응답 등과 같은 다운스트림 태스크에 맞게 파인 튜닝한다.

## 사전 학습된 BERT 모델을 특징 추출기로 사용하는 방법

### 데이터셋의 문장을 벡터화하는 방법

**데이터셋(텍스트)** → 모델에 입력하기 위해 **텍스트를 벡터화**

- 문맥 독립 임베딩 모델
  - TF-IDF, Word2Vec, FastText, GloVe
- 문맥 임베딩 모델
  - ELMo, BERT, GPT-3

### 사전 학습된 BERT에서 임베딩 추출하는 방법

- 단어 임베딩(벡터 표현) 추출
- 문장 임베딩(벡터 표현) 추출

#### 단어 임베딩 추출

사전 학습된 BERT 모델을 사용해 데이터셋의 문장을 벡터화하는 방법

1. 워드피스 토크나이저를 이용해 문장을 토큰화하여 토큰(단어)을 얻는다.
2. 토큰 리스트 시작 부분에 `[CLS]` 토큰을 추가하고 끝에 `[SEP]` 토큰을 추가한다.
3. 모든 토큰의 길이는 동일하게 유지해야 하므로, `[PAD]` 토큰을 이용하여 토큰의 길이를 동일하게 맞춰준다.
4. 어텐션 마스크를 사용하여 `[PAD]` 토큰이 실제 토큰의 일부가 아니라는 것을 모델이 이해하도록 한다.
   - 모든 위치에서 어텐션 마스크값을 1로 설정하고 `[PAD] ` 토큰이 있는 위치에만 0을 설정한다.
5. 모든 토큰을 고유한 토큰 ID에 매핑한다.
6. 사전 학습된 BERT 모델에 대한 입력으로 어텐션 마스크와 함께 토큰 ID를 공급하고 각 토큰의 벡터 표현(임베딩)을 얻는다.

​	위와 같은 방법으로 학습셋의 모든 문장을 벡터화하여 문장에서 각 단어에 대한 표현을 얻는다.

<img src="https://miro.medium.com/max/1400/1*7TdPa1j3HenGRMNSzlIheg.png" width="75%" /> 

*출처 [Understanding the Bert Model](https://medium.com/analytics-vidhya/understanding-the-bert-model-a04e1c7933a9)*

#### 문장 임베딩 추출

`[CLS]` 토큰의 표현은 전제 문장의 집계 표현을 보유하게 되므로, **문장의 표현**은 `[CLS]` 토큰에 해당하는 `R_[CLS]` 표현 벡터가 된다.

- `[CLS]` 토큰의 표현을 문장 표현으로 사용하는 것이 항상 좋은 생각은 아니다. 문장의 표현을 얻는 효율적인 방법은 모든 토큰의 표현을 평균화하거나 풀링하는 것이다. _Cf. 4장_ 

- 매우 유사한 방식으로 학습셋에 있는 모든 문장의 벡터 표현을 계산하여 모든 문장의 문장 표현을 얻은 후에는 해당 표현을 입력으로 제공하고 분류기로 학습해 감정 분석 작업을 수행할 수 있다.

### Hugging Face Transformer

Hugging Face의 오픈 소스 라이브러리 `transformers`와 사전 학습된 BERT 모델을 사용해, 문장에 있는 모든 단어의 문맥화된 단어 임베딩 생성하기

- Hugging Face : 자연어 기술의 민주화를 추구하는 조직
- `transformers` : 파이토치 및 텐서플로와 모두 호환, NLP 및 NLU 태스크에 강력, 100개 이상의 언어로 사전 학습된 수천 개의 모델 포함

#### BERT의 최상위 인코더 계층인 12번째 인코더에서만 임베딩을 얻는 방법

- 💻 [최종 인코더 계층에서만 임베딩을 추출하는 실습 코드](../../codes/3-2_generating_BERT_embedding.ipynb) *Cf. Google Colab / Python 3.x*

- 코드 흐름

  1. 설치 및 다운로드
     -  `transformers` 설치
     - 사전 학습된 BERT *model* 다운로드
     - 위 모델을 사전 학습시키는 데 사용된 *tokenizer* 다운로드 

  2. 입력 전처리

     - *tokenizer*를 이용해 문장 토큰화
     - *attention_mask* 생성
     - *token_ids*로 변환

  3. 임베딩 추출

     - *token_ids* 및 *attention_mask*를 *model*에 입력하고 임베딩 획득

     - *model*은 두 값으로 구성된 튜플 반환

       ```
       last_hidden_state, pooler_output = model(token_ids, attention_mask = attention_mask)
       ```

       `last_hidden_state` : 최종 인코더 계층(12번째 인코더)에서만 얻은 모든 토큰의 표현 벡터

        `pooler_output` : 최종 인코더 계층의 `[CLS]` 토큰 표현을 나타내며 선형 및 tanh 활성화 함수에 의해 계산

       📌 `last_hidden_state[0][0]` vs. `pooler_output`
       
       - https://github.com/huggingface/transformers/issues/7540
       
       *Cf. 책에서는 last_hidden_state를 hidden_rep로, pooler_output을 cls_head라고 기술했다.*

#### BERT의 모든 인코더 레이어에서 임베딩을 추출하는 방법

- 최종 인코더 레이어(마지막 계층의 은닉 상태)에서만 얻은 임베딩(표현 벡터) 사용 **vs.** 다른 인코더 레이어에서 얻은 임베딩 고려

  ➡︎ 모든 인코더 레이어(모든 은닉 상태)에서 얻은 임베딩도 고려하자!

|              속성              |      표기       | F1 스코어 |
| :----------------------------: | :-------------: | :-------: |
|             임베딩             |       h_0       |   91.0    |
|   마지막에서 두 번째 레이어    |      h_11       |   95.6    |
|         마지막 레이어          |      h_12       |   94.9    |
|   마지막 4개 레이어의 가중합   |   h_9 to h_12   |   95.9    |
| **마지막 4개 레이어의 연결값** | **h_9 to h_12** | **96.1**  |
|      12개 레이어의 가중합      |   h_1 to h_12   |   95.5    |

<div style="text-align:center">구글 BERT의 정석 그림 3-5 서로 다른 레이어의 임베딩 속성으로 도출한 F1 스코어<div>

- 💻 [모든 인코더 레이어에서 임베딩 추출하는 실습 코드](../../codes/3-3_extracting_embeddings_from_all_encoder_layers_of_BERT.ipynb) *Cf. Google Colab / Python 3.x*

- BERT의 최상위 인코더 계층인 12번째 인코더에서만 임베딩을 얻는 코드와 유사하고, 다른 점은 다음과 같다 :

  - 사전 학습된 BERT 모델을 다운로드 받을 때

     모든 인코더 레이어에서 임베딩을 얻기 위해,  `output_hidden_states = True` 로 설정한다.

  - 임베딩을 가져올 때

    모델은 hidden_state를 추가하여, 2개가 아닌 3개의 값이 있는 튜플을 반환한다.
    
    ```
    last_hidden_state, pooler_output, hidden_states = model(token_ids, attention_mask = attention_mask)
    ```
    
    `last_hidden_state` 와 `pooler_output` 의 값은 최상위 인코더 계층에서만 임베딩을 얻는 경우와 동일하고, `hidden_states`가 추가된다.
    
    `hidden_states` : 모든 인코더 계층에서 얻은 모든 토큰의 표현 포함
    
    - 입력 임베딩 레이어 *h_0*에서 *h_12*까지 모든 인코더 레이어의 표현을 포함하는 13개의 값을 포함하는 튜플
    
      *hidden_states[i]는 i번째 레이어 h_i에서 얻은 모든 토큰의 표현 벡터를 가진다. => hidden_states[12]==last_hidden_state*

## 정리

- 사전 학습된 BERT 모델을 다음 두 가지 방법으로 사용할 수 있다.

  - 임베딩을 추출해 특징 추출기로 사용한다. 

  - 사전 학습된 BERT 모델을 다운스트림 태스크에 맞게 파인 튜닝한다.

  _Cf. 이 문서에서는 임베딩을 추출하는 방법만 다뤘다._

- 사전 학습된 BERT에서 단어 및 문장 임베딩(표현)을 추출할 수 있다.

  - 단어의 벡터 표현
    - 최종 인코더는 문장에 있는 모든 토큰(단어)의 최종 표현 벡터(임베딩)을 반환한다.
  - 문장의 벡터 표현
    - `[CLS]` 토큰의 표현은 전체 문장의 집계 표현을 보유하게 된다. 

- 사전 학습된 BERT 모델에서 임베딩을 추출할 때 어떤 레이어에서 얻은 임베딩을 사용해야 할까?

  - 최종 인코더 레이어(마지막 계층의 은닉 상태)에서만 얻은 임베딩
  - 모든 인코더 레이어(모든 은닉 상태)에서 얻은 임베딩

- 실습 코드

  - 저자의 깃허브 저장소
    -  [3.03. Generating BERT embedding .ipynb](https://github.com/PacktPublishing/Getting-Started-with-Google-BERT/blob/main/Chapter03/3.03.%20Generating%20BERT%20embedding%20.ipynb)
    - [3.04. Extracting embeddings from all encoder layers of BERT.ipynb](https://github.com/PacktPublishing/Getting-Started-with-Google-BERT/blob/main/Chapter03/3.04.%20Extracting%20embeddings%20from%20all%20encoder%20layers%20of%20BERT.ipynb)

  - 버전 업데이트를 반영하고, 한글 설명을 추가한 코드 *Cf. Google Colab / Python 3.x*

    - [최종 인코더 계층에서만 임베딩 추출](../../codes/3-2_generating_BERT_embedding.ipynb)

    - [모든 인코더 레이어에서 임베딩 추출](../../codes/3-3_extracting_embeddings_from_all_encoder_layers_of_BERT.ipynb)

## 참고 자료

- 구글 BERT의 정석 [book](http://www.yes24.com/Product/Goods/104491152)
