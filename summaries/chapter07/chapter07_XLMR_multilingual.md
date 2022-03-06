# Chapter 7 다른 언어에 BERT 적용하기

## XLM-R 이해하기

- XLM에서 몇 가지를 보완한 확장 버전
- XLM-RoBERTa
- 자료가 적은 언어의 경우 병렬 데이터셋을 구하는 것이 쉽지 않아 MLM만 학습
- 커먼 크롤 데이터셋에서 별도 레이블이 없는 100개의 언어 텍스트를 필터링하여 얻은 2.5TB의 데이터셋으로 학습
- 비중이 적은 언어는 oversampling
- Common crawl이 Wikipedia에 비해 자료가 적은 언어에서 데이터가 더 많음
- Sentencepiece Tokenizer, 25만 개의 Token
- Model Architecture
  - XLM-R_base: 12개의 Encoder layers, 12개의 Attention heads, 768 Hidden size
  - XLM-R: 24개의 Encoder layers, 16개의 Attention heads, 1024 Hidden size
- M-BERT와 XLM보다 더 뛰어난 성능
- XLM-R의 교차 언어 분류 태스크로 모델을 평가
- 15개의 서로 다른 XNLI 데이터셋으로 평가
- 가장 낮은 점수인 스웨덴어에서 M-BERT 50.4% → XLM-R 73.9%
- XLM-R의 평균 정확도가 80.9%로 다른 모델보다 비교적 높음

## 언어별 BERT

- 여러 언어 대신 특정 단일 언어만 이용해 BERT 학습

### 프랑스어 FlauBERT

- 사용 말뭉치: Wikipedia, 도서, 내부 크롤링, WMT19, OPUS(오픈 소스 병렬 코퍼스)의 프랑스어 텍스트 등의 24개
- Moses Tokenizer: URL, 날짜 등을 포함한 특수 토큰 보존
- 전처리 및 토큰화 후 BPE 사용, vocab building
- 50,000개의 Token
- MLM만 수행, Dynamic masking 사용
- small-cased, base-uncased, base-cased, large-cased
- HuggingFace에서 사용 가능
- FLUE(French Language Understanding Evaluation)
  - CLS-FR
  - PAWS-X-FR
  - XNLI-FR
  - French Treebank
  - FrechSemEval

### 스페인어 BETO

- WWM(Whole Word Masking) + MLM
- POS, NER-C, MLDoc, PAWS-X, XNLI로 테스트
- HuggingFace에서 사용 가능

```python
from transformers import pipeline

predict_mask = pipeline(
		"fill-mask",
		model = "dccuchile/bert-base-spanish-wwm-uncased",
		tokenizer = "dccuchile/bert-base-spanish-wwm-uncased"
)

result = predict_mask('[MASK] los caminos llevan a Roma')
```

### 네덜란드어 BERTje

- WWM과 MLM 및 SOP를 동시에 진행
- 사용 말뭉치: TwNC(네덜란드 뉴스 말뭉치), SoNAR-500(다중 장르 참조 말뭉치), 네덜란드 위키피디아 텍스트, 웹 뉴스 및 서적
- 100만 번의 iteration으로 학습
- HuggingFace에서 사용 가능

### 독일어 BERT

- Cloud TPU v2에서 9일 동안 Wikipedia text, 뉴스, OpenLegalData

### 중국어 BERT

- 12개의 Encoder layers, 12개의 Attention heads, 768개의 hidden unit, 110M
- WWM + MLM
  - WWM을 사용해 pretrain → 하위 단어가 masking 되면 하위 단어를 포함하는 전체 단어를 마스킹
- LTP(Language Technology Platform)를 사용
  - LTP는 단어 분할, 형태소 분석, 구문 분석을 수행하는데 사용
  - 중국어 단어 경계를 식별하는 데도 이용

### 일본어 BERT

- 일본어 Wikipedia를 사용해 WWM으로 학습
- MeCab으로 Tokenization → Wordpiece Tokenizer로 subword를 얻음

### 핀란드어 FinBERT

- M-BERT보다 성능이 좋음
- Wikipedia에서 핀란드 텍스트의 비중은 3%에 불과
- FinBERT는 핀란드어 뉴스 기사, 온라인 토론 및 인터넷 크롤링 텍스트로 학습
- FinBERT-cased, FinBERT-uncased
- WWM을 이용해 MLM과 NSP로 Pretrained

### 이탈리아어 UmBERTo

- RoBERTa 아키텍쳐를 따름
  - MLM Task시에 Dynamic Masking 사용
  - NSP 제거 MLM만 사용
  - 큰 batch size 사용
  - Byte-level BPE Tokenizer
- RoBERTa에 Sentencepiece + WWM 사용

### 포르투갈어 BERTimbau

- brWaC: 포르투갈어 대규모 오픈 소스 말뭉치
- WWM + MLM 100만 iteration

### 러시아어 RuBERT

- M-BERT에서 Knowledge Distillation
- 학습 전에 Word Embedding을 제외하고 RuBERT의 변수를 M-BERT 모델의 변수로 초기화
- 러시아어 Wikipedia text와 뉴스 기사를 사용해 학습
- Subword NMT로 텍스트를 subword로 나누는 데 사용
- RuBERT의 Subword vocab은 M-BERT의 vocab에 비해 더 길고 더 많은 러시아어롤 구성
- M-BERT와 RuBERT 어휘 모두에서 나타나는 일반적인 단어는 M-BERT의 임베딩으로 직접 사용 가능