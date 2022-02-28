# Chapter 6 텍스트 요약을 위한 BERTSUM 탐색

BERTSUM: 텍스트 요약에 맞춰 파인 튜닝된 BERT 모델

## BERTSUM 모델의 성능

BERTSUM 모델의 성능은 ROUGE 점수를 사용하여 평가

### ROUGE 평가 지표

 **ROUGE(Recall-oriented Understudy for Gisting Evalutation)**

- 텍스트 요약과 기계 번역 태스크를 평가하는 데 사용하는 평가 지표
- 「[ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/)」

<u>ROUGE의 다섯 가지 평가 지표 형태</u>

- ROUGE-N
- ROUGE-L
- ROUGE-W
- ROUGE-S
- ROUGE-SU

*Cf. [ROUGE-an Evaluation Metric for Text Summarization](https://ilmoirfan.com/rouge-an-evaluation-metric-for-text-summarization/)*

#### ROUGE-N

후보 요약(예측한 요약)과 참조 요약(실제 요약) 간의 n-gram 재현율(recall)

```
재현율 = (예측한 요약 결과와 실제 요약 사이의 서로 겹치는 n-gram 총 수) / (실제 요약의 n-gram의 총 수)
```

*ROUGE-1(uni-gram)과 ROUGE-2(bi-gram)가 가장 많이 쓰인다.*

#### ROUGE-L

**가장 긴 공통 하위 시퀀스(LCS)**를 기반으로 하며, F-measure를 사용해 측정

### BERTSUM을 사용한 요약 태스크의 ROUGE 점수

- 분류기, 트랜스포머, LSTM을 BERT에 적용한 BERTSUM 모델을 사용한 **추출 요약** 태스크의 ROUGE 점수 

  *[Yang Liu. 2019. Fine-tune BERT for Extractive Summarization](https://arxiv.org/abs/1903.10318) 에서 CNN/DailyMail 테스트 데이터를 이용해 측정한 결과 (25/3/2019)*

  *Cf. https://github.com/nlpyang/BertSum*

  |          모델           |  ROUGE-1  |  ROUGE-2  |  ROUGE-L  |
  | :---------------------: | :-------: | :-------: | :-------: |
  |  Transformer Baseline   |   40.9    |   18.02   |   37.17   |
  |   BERTSUM+classifier    |   43.23   |   20.22   |   39.60   |
  | **BERTSUM+transformer** | **43.25** | **20.24** | **39.63** |
  |      BERTSUM+LSTM       |   43.22   |   20.17   |   39.59   |

- BERTSUMABS 모델로 **생성 요약** 태스크를 수행했을 때의 ROUGE 점수	

  *[Yang Liu, Mirella Lapata. 2019. Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345) 에서 CNN/DailyMail 테스트 데이터를 이용해 측정한 결과 (20/8/2019)*

  *Cf. https://github.com/nlpyang/PreSumm*

  |    모델    | ROUGE-1 | ROUGE-2 | ROUGE-L |
  | :--------: | :-----: | :-----: | :-----: |
  | BERTSUMABS |  41.72  |  19.39  |  38.76  |

## BERTSUM 모델 학습

```python
# Google Colab / Python 3.x / GPU

%%capture
!pip install pytorch_pretrained_bert
!pip install torch==1.1.0 pytorch_transformers tensorboardX multiprocess pyrouge
!pip install googleDriveFileDownloader

cd /content/

# BERTSUM 연구원들이 오픈 소스로 제공한 학습 코드
!git clone https://github.com/nlpyang/BertSum.git
  
# 데이터셋 다운로드 (전처리된 CNN/DailyMail 뉴스 데이터)
cd /content/BertSum/bert_data/
### 다운로드 및 압축 해제 ###

# BERTSUM 모델 학습
cd /content/BertSum/src

'''
First run: For the first time, you should use single-GPU, so the code can download the BERT model. Change -visible_gpus 0,1,2 -gpu_ranks 0,1,2 -world_size 3 to -visible_gpus 0 -gpu_ranks 0 -world_size 1, after downloading, you could kill the process and rerun the code with multi-GPUs.
'''

# BERTSUM + classifier (number of parameters: 109483009)
!python train.py -mode train -encoder classifier -dropout 0.1 -bert_data_path ../bert_data/cnndm -model_path ../models/bert_classifier -lr 2e-3 -visible_gpus 0 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 50 -accum_count 2 -log_file ../logs/bert_classifier -use_interval true -warmup_steps 10000

# BERTSUM + transformer (* number of parameters: 120512513)
!python train.py -mode train -encoder transformer -dropout 0.1 -bert_data_path ../bert_data/cnndm -model_path ../models/bert_transformer -lr 2e-3 -visible_gpus 0 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 50 -accum_count 2 -log_file ../logs/bert_transformer -use_interval true -warmup_steps 10000 -ff_size 2048 -inter_layers 2 -heads 8

# BERTSUM + lstm (* number of parameters: 113041921)
!python train.py -mode train -encoder rnn -dropout 0.1 -bert_data_path ../bert_data/cnndm -model_path ../models/bert_rnn -lr 2e-3 -visible_gpus 0 -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 50 -accum_count 2 -log_file ../logs/bert_rnn -use_interval true -warmup_steps 10000 -rnn_size 768 -dropout 0.1
```

```
usage: train.py [-h] [-encoder {classifier,transformer,rnn,baseline}]
                [-mode {train,validate,test}] [-bert_data_path BERT_DATA_PATH]
                [-model_path MODEL_PATH] [-result_path RESULT_PATH]
                [-temp_dir TEMP_DIR] [-bert_config_path BERT_CONFIG_PATH]
                [-batch_size BATCH_SIZE] [-use_interval [USE_INTERVAL]]
                [-hidden_size HIDDEN_SIZE] [-ff_size FF_SIZE] [-heads HEADS]
                [-inter_layers INTER_LAYERS] [-rnn_size RNN_SIZE]
                [-param_init PARAM_INIT]
                [-param_init_glorot [PARAM_INIT_GLOROT]] [-dropout DROPOUT]
                [-optim OPTIM] [-lr LR] [-beßta1 BETA1] [-beta2 BETA2]
                [-decay_method DECAY_METHOD] [-warmup_steps WARMUP_STEPS]
                [-max_grad_norm MAX_GRAD_NORM]
                [-save_checkpoint_steps SAVE_CHECKPOINT_STEPS]
                [-accum_count ACCUM_COUNT] [-world_size WORLD_SIZE]
                [-report_every REPORT_EVERY] [-train_steps TRAIN_STEPS]
                [-recall_eval [RECALL_EVAL]] [-visible_gpus VISIBLE_GPUS]
                [-gpu_ranks GPU_RANKS] [-log_file LOG_FILE] [-dataset DATASET]
                [-seed SEED] [-test_all [TEST_ALL]] [-test_from TEST_FROM]
                [-train_from TRAIN_FROM] [-report_rouge [REPORT_ROUGE]]
                [-block_trigram [BLOCK_TRIGRAM]]
```

## 참고 자료

- 구글 BERT의 정석 [book](http://www.yes24.com/Product/Goods/104491152)
- [Fine-tune BERT for Extractive Summarization (BERTSUM) github repository](https://github.com/nlpyang/BertSum)
- [ROUGE-an Evaluation Metric for Text Summarization](https://ilmoirfan.com/rouge-an-evaluation-metric-for-text-summarization/)

 