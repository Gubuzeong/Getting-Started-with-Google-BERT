# Chapter 6: 텍스트 요약을 위한 BERTSUM 탐색

> Speaker: 남수연


# 텍스트 요약

NLP 분야의 주요 연구 분야 중 하나로, 주어진 긴 텍스트를 요약하는 것. 긴 문서, 뉴스 기사, 법률 문서, 블로그 게시물 등 다양한 영역에서 널리 사용됨. 

# 텍스트 요약 방식 이해하기

아래와 같은 텍스트를 요약해야 한다고 해보자.

```
나는 어제 신촌에서 동아리 운영진 동기 언니와 10시간 내내 먹었다. 점심으로
진돈부리를 가려고 했지만 딱 어제 휴업하는 바람에 반서울에 갔는데 엄청 맛있었다.
다음에 또 와야겠다고 생각했다. 후식으로 파이홀에 가서 오레오말차가나슈파이와 얼그
레이가나슈파이를 먹었다. 역시 다음에 또 와야겠다고 생각했다. 저녁으로 돈우마미에
가서 사케동을 먹었다. 가라아게 4조각을 시켰는데 서비스로 한 조각을 더 주셔서
돈우마미는 참 좋은 가게라는 생각이 들었다. 마지막으로 아워즈에 가서 칵테일을 조
졌다. 줄리엣이라는 칵테일을 주문했는데 요맘때 복숭아맛과 딸기맛을 섞어놓은 것 같은
것이 정말 정말 맛있었다. 정신을 차려보니 9시 30분이어서 허겁지겁 버스를 타고
집에 왔더니 월요일이 스터디 하는 날이랜다. 어이가 없어서 헛웃음이 나왔지만 지금
웃을 때가 아니므로 열심히 스터디 준비를 하는 중이다.
```

## 추출 요약

주어진 텍스트 안에서 중요한 문장만 추출해 요약한다. 입력의 문장 중 중요도가 높은 순으로 N개의 문장을 뽑는 식이다.

```
나는 어제 신촌에서 동아리 운영진 동기 언니와 10시간 내내 먹었다. 점심으로
진돈부리를 가려고 했는데 딱 어제만 휴업하는 바람에 반서울에 갔는데 엄청 맛있
었다. 후식으로 파이홀에 가서 오레오말차가나슈 파이와 얼그레이가나슈파이를 먹었다.
마지막으로 아워즈에 가서 칵테일을 조졌다. 어이가 없어서 헛웃음이 나왔지만 지금
웃을 때가 아니므로 열심히 스터디 준비를 하는 중이다.
```

## 생성 요약

주어진 텍스트를 의역해서 원래 문서에 포함되지 않은 문장으로 요약을 만든다.

```
어제의 나는 동아리 언니와 10시간동안 쉬지 않고 먹었다. 정신을 차려보니 발등에
불이 떨어진 상태여서 열심히 스터디 준비를 하는 중이다.
```

## Summary

|  | 추출 요약 | 생성 요약 |
| --- | --- | --- |
| 입력 문장을 그대로 사용하는가? | O | X |
| 정보 누락 정도 | 많음 | 적음 |
| 컴퓨팅 리소스 필요량 | 적음 | 많음 |
| 학습 시간 필요량 | 적음 | 많음 |

# 텍스트 요약에 맞춘 BERT 파인 튜닝

## BERT을 활용한 추출 요약

사전학습된 BERT를 사용해 추출 요약 태스크를 진행하려면 BERT 모델 입력 데이터 형태를 수정해야 한다. BERT 모델의 입력 데이터를 수정하여 각 문장의 표현을 얻는 방법에 대해 알아보자.

- BERT 복습
    1. 입력 문장을 토큰 형태로 변경한다.
    2. 첫 문장의 시작 부분에만 `[CLS]` 토큰을 모든 문장의 마지막 부분에 `[SEP]` 토큰을 추가한다. 
    3. 입력 토큰을 토큰 임베딩, 세그먼트 임베딩, 위치 임베딩, 총 3개의 임베딩 레이어 형태로 각각 변환한다.
    4. 모든 임베딩을 더한 다음 BERT에 입력한다.
    5. BERT는 아웃풋으로 모든 토큰의 표현 벡터를 출력한다.
    


### 1) 토큰 임베딩

텍스트 요약 태스크에서는 BERT 모델에 여러 문장을 입력하고 입력한 모든 문장에 대한 표현이 필요하다. **모든 문장의 시작 부분에 `[CLS]` 토큰을 추가**하면 모든 문장에 대한 표현을 얻을 수 있다. 이 `[CLS]` 토큰은 각 문장을 대표하는 토큰으로서 문장의 특징을 뽑아낼 때 사용할 것이다.

```python
input = ["[CLS]", "ban", "seoul", "[SEP]", "[CLS]", "pie", "hole", "[SEP]", "[CLS]", "don", "umami", "[SEP]"]
```

### 2) 세그먼트 임베딩

세그먼트 임베딩은 입력을 $E_A, E_B$ 형태로 반환한다. 그보다 훨씬 많은 수의 문장이 입력으로 들어오는 텍스트 요약 태스크에서는  $E_A, E_B$ 만으로 어떻게 문장을 구분할 수 있을까?

인터벌 세그먼트 임베딩을 통해 홀수번째 문장에서 발생한 토큰은 $E_A$, 짝수번째 문장에서 발생한 토큰은 $E_B$에 매핑한다. 한 문장을 앞뒤의 문장들과 구분하기만 하면 되므로 이런 방법을 사용할 수 있다.

### 3) 위치 임베딩

위치 임베딩은 모든 토큰의 위치 정보에 대한 임베딩값으로, 기존 방식과 동일하다. 

## BERTSUM

![Untitled](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdWD6D4%2FbtruAsTYw1i%2Fjz2Wj3OWpG8nNTIO6oKzvk%2Fimg.png)

BERT 모델을 사용해 입력 데이터 형식을 변경해서 텍스트 요약에 특화되도록 만든 모델을 BERTSUM이라고 한다. 처음부터 BERT를 학습시킬 필요는 없으며, 이미 사전학습된 BERT 모델에 앞서 설명한 입력 데이터 형태로 변경하여 파인 튜닝하면 된다.

### 1) 분류기가 있는 BERTSUM


**모든 문장의 표현을 받아서 각 문장의 중요도를 판단하는 분류기 레이어**를 ****BERT에 이어붙인다. 분류기는 각 문장을 요약에 포함시킬지 여부에 대한 확률을 제공한다. Linear Layer 하나만 사용하며, BERT의 아웃풋을 분류기 레이어에 넣어 나온 값에 sigmoid 활성화 함수를 취한 값을 target으로 활용한다. 이렇게 계산된 target과 정답을 BCE를 통해 loss를 계산한다. loss 값을 최소화하도록 BERT 모델과 분류기 레이어를 함께 학습시킨다.

$$
\hat{Y} = \sigma(W_oT_i + b_o)
$$

### 2) 🌟 트랜스포머와 LSTM을 활용한 BERTSUM

**(1) 문장 간 트랜스포머(inter-sentence transformer)를  활용한 BERTSUM**

논문에 따르면 다른 2개의 Summarization layer보다 훨씬 좋은 결과를 보여주는 방법이다. BERT의 결과인 문장 표현 $R$을 트랜스포머 레이어에 입력한다. 트랜스포머는 BERT에서 얻은 표현을 가져와 은닉 상태로 변환하는데, 이 때 도입되는 트랜스포머는 문장 간 Attention을 계산하고 문장 단위가 아닌 전체 문서 관점에서 요약 태스크를 수행한다.

BERT에서 얻은 문장 표현 R에 위치 임베딩 값을 추가하여 트랜스포머의 인코더에 입력한다($h^0=PosEmb(R)$) . 인코더 $l$에서 서브레이어는 다음과 같이 표현한다.

![Untitled](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbJZYZH%2FbtruqEIBzAT%2FeQJ7thxXCv8w6aY1diusEk%2Fimg.png)

최상위 인코더를 $L$, 최상위 인코더에서 나온 은닉 상태를 $h^L$이라고 했을 때 문장의 포함 여부를 계산하는 식은 다음과 같다.

![Untitled](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdI66pR%2FbtruBcXEI0K%2Fkj0ac7WwZqUg4QeDwa4uRK%2Fimg.png)

**(2) LSTM을 활용한 BERTSUM**

BERT의 last hidden layer에 RNN을 활용하면 성능이 좋을 수 있다는 논문을 참고하여 실험을 진행했으며, 훈련 과정을 안정화하기 위해 단순 LSTM이 아닌 pergate layer nomarlization을 사용했다.

BERT에서 얻은 문장 $i$에 대한 표현 $R_i$를 LSTM에 입력하면 LSTM 셀은 은닉 상태 $h_i$를 출력한다. sigmoid에 $h_i$를 입력하면 각 문장을 요약에 포함시킬지에 대한 확률을 반환한다.

![Untitled](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdW6YJz%2Fbtruvt0szCk%2FT0z6gVFpvQJCy03aX9iI61%2Fimg.png)

### 참고: summarization layers에 대한 ROUGE 성능 평가

![스크린샷 2022-02-27 오후 5.36.40.png](https://cdn-images-1.medium.com/max/720/1*LMmlre3d3_M5MbAYTccu5A.png)

# References

[[논문 리뷰] Fine-tune BERT for Extractive Summarization](https://medium.com/@eyfydsyd97/bert%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%ED%85%8D%EC%8A%A4%ED%8A%B8-%EC%9A%94%EC%95%BD-text-summary-b582b5cc7d)

[BERT를 활용한 한국어 문서 추출요약 봇](https://velog.io/@raqoon886/KorBertSum-SummaryBot)
