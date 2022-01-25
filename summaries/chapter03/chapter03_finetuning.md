# 다운스트림 태스크를 위한 BERT 파인 튜닝 방법

처음부터 학습하는 것이 아니라 Pretrained Model을 사용하여 학습

- 텍스트 분류
- 자연어 추론(NLI)
- 개체명 인식(NER)
- 질문-응답

## 텍스트 분류

- Tokenization(Special Tokens 추가)
- R_[CLS]를 문장의 표현으로 사용
- Classifier(softmax)에 넣고 감정 분석

Fine-tuning 하면서 Model Weight를 조절하는 방법은 두 가지

1. 분류 계층과 함께 사전 학습된 BERT 모델의 가중치를 업데이트
2. Pretrained BERT가 아니라 Classifier weight만 업데이트. Feature Extractor로 사용(Model Freeze)

Feature Extractor로 사용하는 것과 무엇이 다를까? Model Free

### 감정 분석을 위한 BERT 파인 튜닝

IMDB Dataset은 영화 리뷰 텍스트와 리뷰의 감정 레이블로 구성

```python
from transfomers import BertForSequenceClassification, BertTokenizerFast, Trainer,
TrainingArguments
from nlp import load_dataset
import torch
import numpy as np

# Data loading
# download from GDrive
dataset = load_dataset('csv', data_files='./imdbs.csv', split='train')
type(dataset) # nlp.arrow_dataset.Dataset
dataset = dataset.train_test_split(test_size=0.3)

train_set = dataset['train']
test_set = dataset['test']

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# ---------------
tokens = [ '[CLS]', 'I', 'love', 'Paris', '[SEP]' ]
inputs_ids = [101, 1045, 2293, 3000, 102]
token_type_ids = [0, 0, 0, 0, 0] #Segment Embedding
attention_mask = [1, 1, 1, 1, 1]

tokenizer('I love Paris') # Same as above

tokenizer(['I love Paris', 'birds fly', 'snow fall'], padding=True, max_length=5)
# ---------------
def preprocess(data):
	return tokenizer(data['text'], padding=True, truncation=True)

train_set = train_set.map(preprocess, batched=True, batch_size=len(train_set))
test_set = test_set.map(preprocess, batched=True, batch_size=len(test_set))

train_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

batch_size = 8
epochs = 2

warmup_steps = 500
weight_decay = 0.01

training_args = TrainingArguments(
	output_dir='./results',
	num_train_epochs=epochs,
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	warmup_steps=warmup_steps,
	weight_decay=weight_decay,
	evaluate_during_training=True,
	logging_dir='./logs',
)

trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=train_set,
	eval_dataset=test_set
)

trainer.train()
trainer.evaluate()
```

## 자연어 추론

- Natural Language Inference
- 가정이 주어진 전제에 참인지 거짓인지 중립인지 여부를 결정하는 태스크
- (전제, 가설)에 대한 레이블이 주어짐
- 전제와 가설 사이를 `[SEP]`으로 구분
- `[CLS]` 토큰의 임베딩의 Classifier에 입력하면 참, 거짓, 중립일 확률을 반환

## 질문-응답(QA Task)

- 목표: 주어진 질문에 대한 단락에서 답을 추출
- BERT의 입력은 Question-Paragraph Pair
- BERT는 Paragraph에서 Answer에 해당하는 텍스트의 범위를 반환해야 함
- 단락 내 답의 시작과 끝 토큰의 확률을 구하면 답을 추출할 수 있음
- 시작 벡터 S, 끝 벡터 E(Trainable)
- Token 표현 벡터 R과 시작 벡터 S 사이의 내적 계산 → Softmax
- 비슷하게 끝 벡터 E에 대해서도 학습

```python
from transformer import BertForQuestionAnswering, BertTokenizer

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-fine-tuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-fine-tuned-squad')

question = "면역 체계는 무엇입니까?"
paragraph = "면역 체계는 질병으로부터 보호하는 유기체 내의 다양한 생물학적 구조와 과정의 시스템입니다. 제대로 기능하려면 면역 체계가 바이러스에서 기생충에 이르기까지 병원균으로 알려진 다양한 물질을 탐지하고 유기체의 건강한 조직과 구별해야 합니다."

question = '[CLS]' + question + '[SEP]'
paragraph = paragraph + '[SEP]'

question_tokens = tokenizer.tokenize(question)
paragraph_tokens = tokenizer.tokenize(paragraph)

tokens = question_tokens + paragraph_tokens
input_ids = tokenizer.convert_to_ids(tokens)

segment_ids = [0] * len(question_tokens)
segment_ids += [1] * len(paragraph_tokens)

input_ids = torch.tensor([input_ids])
segment_ids = torch.tensor([segment_ids])

start_scores, end_scores = model(input_ids, token_type_ids=segment_ids)

start_index = torch.argmax(start_scores)
end_index = torch.argmax(end_scores)

print(' '.join(tokens[start_index:end_index+1]))
```

## 개체명 인식(NER)

- 개체명을 미리 정의된 범주로 분류
- 앞에 `[CLS]` 끝에 `[SEP]` 토큰 추가
- BERT 모델에 토큰을 입력하고 표현 벡터 반환
- Classifier에 입력
- Classifier는 개체명이 속한 범주를 반환

# 텐서플로 2와 머신러닝으로 시작하는 자연어 처리

- LM이란 단어들의 시퀀스에 대한 확률 분포
- 단어들의 모임(시퀀스)가 있을 때 해당 단어의 모음이 어떤 확률로 등장하는지를 나타내는 값
- BERT Fine tuning 예시
  - 언어적 용인 가능성(Linguistic Acceptability)
  - 자연어 추론(Natural Language Inference)
  - 유사도 예측(Similiarity Predicition)
  - 감정 분석(Sentiment Analysis)
  - 개체명 인식(Named Entity Recognition)
  - 기계독해(Reading Comprehension)
- BERT 파생모델 분류
  1. BERT를 개선하려고 노력한 모델. 성능 향상, 속도 개선, 메모리 최적화 등 a.g.) SpanBERT, RoBERTa, ERNIE
  2. BERT의 알고리즘 문제를 실험적으로 증명하며 개선하려는 모델 a.g.) XLNet
  3. BERT를 NLP가 아닌 다른 분야에 사용하는 모델 a.g.) VideoBERT, VisualBERT

# Questions

1. 현업에서 Fine-tuning 어떻게 하는지?
   - https://klue-benchmark.com/
   - https://huffon.github.io/2019/11/16/glue/
   - https://gluebenchmark.com/
   - https://korquad.github.io/
   - https://blog.naver.com/skelterlabs/222025030327
2. Weight를 freezing 하면 어떤 효과가 있는지?
   - https://raphaelb.org/posts/freezing-bert

# Few-shot, One-shot, Zero-shot

- https://velog.io/@tobigs-gm1/Few-shot-Learning-Survey
- https://www.kakaobrain.com/blog/106

# Fine-tuning 논문, Intermediate task 논문

- https://arxiv.org/pdf/2005.00628.pdf

# References

https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/