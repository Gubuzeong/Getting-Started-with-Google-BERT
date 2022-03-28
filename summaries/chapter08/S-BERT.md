# Sentence-BERT 및 domain-BERT 살펴보기

> Speaker: 남수연(@mori8)
>
> 회사에서 molrae 쓰는 중

## Sentence-BERT

Sentence-BERT는 vanila BERT/RoBERTa를 fine-tuning하여 문장 임베딩 성능을 우수하게 개선한 모델이다. BERT/RoBERTa는 STS 태스크에서도 좋은 성능을 보여주었지만 매우 큰 연산 비용이 단점이었는데, Sentence-BERT는 학습하는 데 20분이 채 걸리지 않으면서 다른 문장 임베딩보다 좋은 성능을 자랑한다.



### 등장 배경

**기존의 BERT로는 large-scale의 유사도 비교, 클러스터링, 정보 검색 등에 많은 시간 비용이 들어간다.**

- BERT로 유사한 두 문장을 찾으려면 두 개의 문장을 한 개의 BERT 모델에 넣어야 유사도가 평가된다.
- 따라서 문장이 10000개 있으면 10C2 번의 연산 후에 유사도 랭킹을 얻을 수 있다.
- 클러스터링이나 검색에서는 각 문장을 벡터 공간에 매핑하는데, BERT를 이용할 때는 단어 표현을 평균내거나 `[CLS]` 토큰의 값을 이용하지만 이랬을 때의 결과는 각 단어의 GloVe 벡터를 평균낸 것보다 나쁘다.



## Sentence-BERT의 문장 임베딩

1. BERT의 `[CLS]` 토큰의 표현 벡터를 문장 표현으로 사용한다.
2. BERT의 모든 단어의 표현 벡터를 평균 풀링하여 만든 벡터를 문장 표현으로 사용한다.
3. BERT의 모든 단어의 표현 벡터를 최대 풀링하여 만든 벡터를 문장 표현으로 사용한다.

Sentence-BERT(이후 SBERT로 표기)는 BERT의 출력에 풀링 연산을 추가한 모델이며, 풀링 방법은 `[CLS]` 토큰의 결과를 사용하는 방법, 모든 출력 벡터를 평균내어 사용, 출력 벡터의 max-over-time*을 계산해 사용하는 방법이 있다. 기본적으로 SBERT는 평균 풀링을 사용하며, 평균 풀링으로 문장 표현을 얻으면 이 표현은 본질적으로 모든 단어의 의미를 갖는다. 반면 최대 풀링으로 문장 표현을 얻을 경우 문장 표현은 본질적으로 중요한 단어의 의미를 갖는다. 

> Max-over-time pooling: 문장의 길이가 다 다르면 문장마다의 feature map 개수가 달라지는데, 모든 문장마다 하나의 값을 갖도록 feature map 벡터 중 가장 큰 값 하나만 사용하는 것



## Fine Tuning 전략: 문장 쌍 분류, 회귀 태스크

신체의 일부를 공유하는 샴 쌍둥이처럼, 샴 네트워크는 두 네트워크가 weight를 공유한다. SBERT는 동일한 사전 학습된 BERT 모델 2개를 사용하여 문장 1의 토큰은 한 BERT로, 문장 2의 토큰은 또 다른 BERT로 입력하고 주어진 문장의 표현을 계산한다. 두 문장을 `[SEP]`으로 구분하여 한 BERT 모델에 같이 집어넣는 게 아니라, 같은 가중치를 갖는 서로 다른 BERT 모델 2개에 각각 넣는 것이다.



## Objective Functions

![img](https://blog.kakaocdn.net/dn/TDfTO/btrpBC1sjIO/1OKnS8Fz0J188aSRSzz0sK/img.png)

SBERT 모델의 구조는 학습 데이터에 따라 다르다. 아래 구조에 따라 목적 함수와 모델 구조를 달리 하였다.

- 두 문장의 출력값인 u, v 그리고 element-wise 차이값인 |u-v|를 concatenate한 후 파라미터를 추가하여 학습한다.
- 실제 inference할 때나, regression 방식의 loss function을 쓸 때는 cosine-similarity를 이용한다. Training할 때, 계산된 cosine similarity와 gold label 간의 MSE를 minimize하는 방식으로 학습했다.



### Classification Objective Function

두 문장이 유사한지(1), 유사하지 않은지(0) 판단하는 태스크를 위한 모델이다. 두 문장 임베딩 *u*와 *v*를 그 차이 ∣*u*−*v*∣와 concat해 3*n* 차원의 텐서를 만들고, 이를 가중치 *W**t*∈R3*n*×*k*에 곱한다. 이를 softmax함수에 넣으면 *k*개 label에 대한 분류 작업이 가능해진다. loss는 cross-entropy로 설정했다.

$$O=softmax(W_t(u,v,∣u−v∣))$$



### Regression Objective Function

회귀 태스크의 목표는 주어진 두 문장 사이의 의미 유사도를 예측하는 것이다. 두 문장의 임베딩($u$, $v$) 사이의 코사인 유사도를 계산한다. 



### Triplet Objective Function

#### Siamese, Triplet의 등장 배경: One-shot

조금의 데이터만으로도 학습할 수 있도록 하는 것이 `Few-shot`, 한 장의 사진만으로 학습하도록 하자는 게 `One-shot`



$$max(∣∣s_a−s_p∣∣−∣∣s_a−s_n∣∣+ϵ,0)$$

anchor/positive/negative 문장 *a*,*p*,*n*에 대해 triplet loss는 모델이 *a*와 *p* 사이 거리가 *a*와 *n*사이 거리보다 작게 하도록 학습시킨다. $s_x$는 문장 *x*의 임베딩이고, ∣∣⋅∣∣은 거리고, *ϵ*은 margin이다. margin은 *s**p*가 *s**n*보다 최소한 *ϵ*만큼 *s**a*에 가깝도록 하는 장치이다. 이 논문에서는 Euclidean 거리를 단위로 *ϵ*=1로 설정했다.



## Evaluation

![img](https://lh3.googleusercontent.com/-kGoALzGpAiQ/X3Gd39d8BOI/AAAAAAAABVQ/1I7rYPheejImavnWQ-M62QLisXIBbazuQCNcBGAsYHQ/w640-h165/tbl3%2Bsenteval.PNG)

- 보다시피 BERT의 output을 그대로 쓰는 건 성능이 별로다. InferSent* - GloVe보다 못한 걸 알 수 있다.
- 제안된 siamese 네트워크 구조와 fine-tuning 메커니즘은 InferSent나 Universal Sentence Encoder를 유의미하게 앞서는 결과를 내었다. SBERT가 좋은 성적을 내지 못한 유일한 데이터셋은 SICK-R이다.
- Universal Sentence Encoder는 뉴스, QA 페이지, 토론 포럼처럼 SICK-R 벤치마킹에 적합한 데이터셋에 학습되었다. 반면 SBERT는 위키피디아와 NLI 데이터에만 학습되었다.
- SRoBERTa는 성능이 좋긴 한데, SBERT에 비해 큰 차이를 보이진 않았다.

> InferSent는 siamse BiLSTM에 max pooling을 적용한 문장 임베딩 모델



## References

- https://velog.io/@ysn003/%EB%85%BC%EB%AC%B8-Sentence-BERT-Sentence-Embeddings-using-Siamese-BERT-Networks
- http://mlgalaxy.blogspot.com/2020/09/sentence-bert-sentence-embeddings-using.html
- https://tyami.github.io/deep%20learning/Siamese-neural-networks/
