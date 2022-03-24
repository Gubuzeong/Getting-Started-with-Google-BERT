# Chapter 10 한국어 언어 모델: KoBERT, KoGPT2, KoBART

- 다국어 모델들은 한국어 데이터의 부족 때문에 좋은 성능을 보이지 못함

## KoBERT

- 한국어 위키피디아에서 500만 개의 문장과 5400만 개의 단어 학습
- Huggingface, MXNet-Gluon, ONNX 등 여러 플랫폼에서 사용 가능
- Sentencepiece를 이용해 Tokenizer 학습
- 8,002개의 vocab size

## KoGPT2

- 한국어에 특화된 GPT-2 모델
- 주어진 텍스트를 기반으로 다음 단어를 잘 예측할 수 있도록 학습됨
- 문장 생성에 최적화
- 125M parameter
- 한국어 위키피디아, 뉴스, 모두의 말뭉치 v1.0, 청와대 국민청원 등 40GB의 한국어 텍스트 사용
- BPE Tokenizer
- 51,200 vocab
- 이모티콘, 이모지도 추가

## KoBART

- text infilling 노이즈 함수만 적용
- 40GB 이상의 한국어 텍스트로 사전 학습
- Denoising AutoEncoder
- 6 encoder layers
- 6 decoder layers
- 16 attention heads
- 768 FFN hidden unit
- 120M parameters