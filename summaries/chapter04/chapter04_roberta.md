# Chatper 4 BERT의 파생 모델 1

이 챕터에서 다루는 BERT의 파생 모델들

- ALBERT
- RoBERTa
- ELECTRA
- SpanBERT

## RoBERTa

- Robustly Optimized BERT pre-training Approach
- BERT가 충분히 학습되지 않았음을 확인
- BERT와의 차이
  - MLM에서 Dynamic Masking 사용
  - NSP 제거
  - 배치 크기 증가
  - BBPE Tokenizer 사용

### Dynamic Masking

- Static Masking
  - BERT에서 Masking은 전처리 단계에서 한 번만 수행
  - Epoch 별로 동일한 Masking을 예측
- Dynamic Masking
  - 문장 10개를 복사
  - 각각의 복사된 문장에 대해 무작위로 15%의 확률로 Masking
  - 40 Epoch이면 1,2,3,4,...,10,1,2,3,...와 같이 학습시킴

### NSP 제거

- 연구원들은 NSP가 BERT에 유용하지 않다는 것을 발견
- Ablation Study
  1. Segment-Pair + NSP: NSP를 사용해서 BERT 학습. 원래 BERT 모델 학습 방법과 유사
  2. Sentence-Pair + NSP: NSP를 사용해서 BERT 학습. 입력은 한 문서의 연속된 부분 또는 다른 문서에서 추출한 문장을 쌍으로 구성
  3. Full Sentences: NSP를 사용하지 않고 BERT 학습. 입력은 하나 이상의 문서에서 지속적으로 샘플링한 결과를 사용. 하나의 문서에서 마지막까지 샘플링하면 다음 문서로 넘어감.
  4. Doc Sentences: NSP를 사용하지 않고 BERT 학습. Full Sentences와 비슷하지만 입력값은 하나의 문서에서만 샘플링한 결과를 입력.
- Results

| Model          | SQuAD 1.1/2.0 | MNLI-m | SST-2 | RACE |
| -------------- | ------------- | ------ | ----- | ---- |
| Segment-Pair   | 90.4/78.7     | 84     | 92.9  | 64.2 |
| Sentence-Pair  | 88.7/76.2     | 82.9   | 92.1  | 63.0 |
| Full-Sentences | 90.4/79.1     | 84.7   | 92.5  | 64.8 |
| Doc-Sentences  | 90.6/79.7     | 84.7   | 92.7  | 65.6 |

- 결론적으로 NSP를 제거하는 것이 성능이 더 좋더라!
- 성능은 Full-Sentences보다 Doc-Sentences가 더 좋지만 RoBERTa에서는 Doc-Sentences의 경우 문서에 따라 배치 크기가 달라지기 때문에 Full-Sentences를 사용하여 학습

### 더 많은 데이터로 학습

- BERT에서 사용한 데이터는 16GB
- RoBERTa에서 사용한 데이터는 160GB

### 큰 배치 크기로 학습

- BERT: 256개 배치로 1M Step Pretrain
- RoBERTa: 8000개 배치로 0.3M Step Pretrain
- 배치 크기를 키우면
  - 학습 속도를 높일 수 있고,
  - 모델의 성능도 향상

### BBPE Tokenizer

- BERT는 Wordpiece Tokenizer 사용
- 캐릭터가 아니라 Byte 형태의 시퀀스 사용
- BERT는 vocab size 30000, BBPE는 50000

```python
from transformers import RobertaConfig, RobertaTokenizer

model = RobertaModel.from_pretrained('roberta-base')
model.config

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
tokenizer.tokenize('It was a great day')
```

### Summary

- RoBERTa는 BERT의 파생모델
- MLM Task로만 학습
- Static Masking
- 큰 Batch size
- BBPE Tokenizer

## References

- https://jeongukjae.github.io/posts/3-roberta-review/
- https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=jjys9047&logNo=221671424019
- https://brunch.co.kr/@choseunghyek/7