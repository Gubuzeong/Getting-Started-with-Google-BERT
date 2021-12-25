# Chapter 1. 트랜스포머 입문
## Transformer?
- RNN, LSTM의 long-term dependency 문제를 보완한 아키텍쳐. 
- Attention 사용 초기에는 RNN, LSTM 기반의 Seq2Seq + Attention 형태로 많이 사용했는데 Transformer는 Self-Attention만을 사용함.
- **Encoder-Decoder** 구조.
  - Encoder: 입력 문장의 표현 방법을 학습
  - Decoder: 표현 방법을 입력받아 원하는 문장을 생성

<img alt="Encoder-Decoder Blocks" src="../../images/chap1_1.png" width="500"/>

## Encoder?
- Transformer에서 Encoder는 여러 개를 쌓아서 사용
- Multi-Head Attention과 Feed Forward

### Attention 복습
https://wikidocs.net/22893

Attention(Q, K, V) = Attention value

1. Query에 대해서 모든 Key와의 유사도 구함
2. 유사도를 Key에 매핑된 Value에 반영
3. 적용된 Value를 모두 더하면 -> Attention value

### Self-Attention


## Reference
- [Attention Is All You Need paper](https://arxiv.org/pdf/1706.03762.pdf)
- [남수연님 발표자료](https://github.com/dsc-sookmyung/2021-DeepSleep-Paper-Review/blob/main/Week7/Attention%20Is%20All%20You%20Need.pdf)
- [딥 러닝을 이용한 자연어처리 입문](https://wikidocs.net/31379)