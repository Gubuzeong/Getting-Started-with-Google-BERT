# Chapter 9.2~

BART에 대해서 알아보고 Hugging Face의 BERT관련 라이브러리를 다룬다.



## BART

BERT는 bidirection encoder로 MASK token을 예측하는데, 이러한 구조로 인해 generation task에서 BERT의 사용이 어렵다. 왜냐하면 각각의 MASK token이 softmax/피드포워드 네트워크를 통해 독립적으로 예측되기 때문이다.

GPT는 autoregressive한 구조로 다음 token을 예측할 수 있지만, BERT처럼 bidirection이 아니다.



### BART Architecture

그래서 두 architecture를 합친다.

![chapter9_BART_architecture](..\..\images\chapter9_BART_architecture.PNG)

BART는 BERT와 GPU 모델의 구조를 결합하였기 때문에, bidirection으로 문장을 살펴보며 attention representation을 얻어 디코더를 통해 seq2seq 구조로 generation을 수행한다.

**이 때문에 MLM과는 달리 여러 task에 대해 응용성이 높다**



BART는 seq2seq transformer 구조를 사용했고, GeLUs 활성화 함수를 사용한다. (파라미터 초기화는 N(0, 0.2)) 인코더와 디코더의 layer 수는 같으며, bsae model은 6 layer, large model은 12 layer를 사용한다. 기존의 트랜스포머 디코더와 동일하게, 디코더의 각 레이어에서는 인코더의 마지막 hidden layer와 cross-attention을 한다.

> self-attention: keys와 values가 queries와 똑같은 임베딩에서 나왔을 때
>
> cross-attention: keys와 values가 queires와 다른 임베딩에서 나왔을 때



### 노이징 기술 (Pretraining BART)

BART에서 알아볼 수 있는 핵심은 noising 기술의 유연함이다. 다양한 변형 noising 기법들을 적용시킬 수 있으며, 이 중에서는 문장의 순서를 바꾸거나, 길이를 변경하는 등의 방법론도 있다. 가장 좋은 성능을 보이는 노이징 기법은 문장의 순서를 랜덤하게 섞고, 임의의 길이의 텍스트를 단일 마스크 토큰으로 교체하는 것이다.

> 모델이 전체적인 문장 길이에 대해 학습해야 하고, 변형된 입력에 많은 집중을 요구하는 효과가 있다고 한다.



![chapter9_BART_architecture](..\..\images\chapter9_BART_noising.PNG)

**토큰 마스킹(Token Masking)**

- 랜덤하게 바뀐 [MASK] 토큰 맞추기

**토큰 삭제(Token Deletion)**

- 토큰이 랜덤하게 제거된다.
- 모델은 어떤 위치의 토큰이 없어졌는지에 대해서도 맞춰야 한다.

**토큰 채우기(Text Infilling)**

- 람다 3을 따르는 포아송 분포에서 span의 길이가 샘플링되고, 각 span은 한개의 [MASK] 토큰으로 치환된다.
- span의 길이가 0일 때도 [MASK] 토큰이 생성된다.
- 모델은 [MASK]에 해당하는 단어들을 맞춰야 한다. 즉 얼마나 많은 토큰이 없어졌는지 예측해야 한다.

**문장 셔플(Sentence Permutation)**

- 문장의 순서를 랜덤으로 섞는다.
- 모델은 섞인 토큰들을 원래의 순서로 배열해야 한다. (XLnet에서 영감)

**문서 회전(Document Rotation)**

- 하나의 토큰이 랜덤하게 선택되고, 해당 토큰이 문서의 시작지점이 된다.
- 모델은 해당 문서의 시작점을 찾아야 한다.



### Fine-tuning BART

번역 테스크?? 몇개의 추가적인 transformer 레이어를 쌓아 올리는 것으로 기계 번역의 새로운 방법론을 제시하기도 하였음. 추가된 레이어는 외국어를 noise가 적용된 영어로 번역하는 것이 학습되며, BART 전반적으로 학습되게 되어 BART를 사전 학습된 target-side 언어 모델로 사용ㅎ나다. WMT 루마니안-영어 벤치마크에서 1.1 BELU만큼 성능이 향상되었다고 한다.



![chapter9_BART_architecture](..\..\images\chapter9_BART_fine-tuning.PNG)



**Sequence Classification Tasks - 시퀀스 단위의 분류 문제**

- 시퀀스 분류 문제. BERT에서 [CLS] 토큰으로 classification task를 수행하는 것에서 영감. BART에서는 디코더에서 마지막 토큰을 추가하여 해당 토큰의 representation이 전체 입력과 attention을 수행할 수 있게 한다. -> output은 모든 입력을 반영

- CoLA: 문장이 문법적으로나 영어적으로 합당한지 분류

**Token Classification Tasks - 토큰 단위의 분류 문제**

- 디코더 가장 위의 hidden state를 각 토큰의 representation으로 사용 -> classification
- SQuAD: 정답에 해당되는 start point, end point 토큰 찾기

**Sequence Generation Tasks - 시퀀스 생성 문제**

- Abstractive QA(question answering), 요약 등의 generation task
- 이러한 테스크는 입력 시퀀스의 내용을 조정하는 특징이 있는데, denoising pre-training objective와 밀접하게 연관되어 있다고 한다.

**Machine Translation**

- 몇개의 추가적인 transformer layer를 쌓아 올려 기계 번역의 새로운 방법론 제시 -> BART 전체를 decoder로 사용
- BART 인코더의 embedding layer를 새롭게 초기화된 인코더로 교체 -> 해당 인코더를 학습시키는 것으로 영어가 아닌 다른 단어들을 영어로 매핑하여 BART가 외국어를 denoise할 수 있게 한다.
- 인코더를 2가지 step으로 학습시킨다.
  - BART 모델의 output에 대한 cross-entropy loss 사용
  - step1) 새롭게 초기화된 인코더, BART의 positional embedding, 첫번째 인코더 layer의 self-attention input projection matrix 학습
  - step2) 모든 모델 parameter를 작은 iteration으로 학습



### Comparing pre-training objectives

실험한 언어 모델의 종류

- Language Model
- Permuted Language Model
- Masked Language Model
- Multitask Masked Language Model
- Masked Seq2Seq



**Tasks**

**SQuAD**

- 위키 문단에 대한 QA task로, 위키피디아에서 가져온 본문과 질문이 주어지면, 주어진 본문으로부터 정답에 해당되는 span을 찾아야 한다. -> BART에서는 start point, end point의 두개를 예측하는 분류기 사용

**MNLI**

- NLI task와 비슷하며, 이전 문장과 이후 문장의 관계 성립 예측. -> BART에서는 두 개의 문장을 [EOS] token으로 합치고, [EOS] 토큰의 representation으로 classification 수행

**[ELI5](https://facebookresearch.github.io/ELI5/)**

- Long form QA task.

**[XSum](https://arxiv.org/pdf/1808.08745v1.pdf)**

- 뉴스 요약 테스크 -> 함축된 요약을 만들어야 한다.

**ConvAI2**

- 대화의 답변 만들기 -> context와 화자가 주어진다.

**CNN/DM**

- 뉴스 요약 테스크 -> 요약본이 입력 문서와 밀접하게 연관되어 있어 XSum과는 약간 다르다.

![chapter9_BART_architecture](..\..\images\chapter9_BART_comparison.PNG)



### Result

- **Performance of pre-training methods varies significantly across tasks**

- **Token masking is crucial**

- **Left-to-right pre-training improves generation**

- **Bidirectional encoders are crucial for SQuAD**

- **The pre-training objective is not the only important factor**

- **Pure language models perform best on ELI5**

- **BART achieves the most consistently strong performance.**





## BERT 라이브러리 탐색

BERT 관련 모델을 쉽게 학습하거나 pre train model을 쉽게 사용 할 수 있는 라이브러리들이 있다.



### ktrain

https://github.com/amaiya/ktrain



### bert-as-service

https://github.com/hanxiao/bert-as-service

