# APPLES_SERchers
Speech Emotion Recognition &amp; Dialogue Emotion Prediction Model
화자 간 감정 정보를 이용한 대화 감정 예측 모델 

## 1. Overview & Research Background 
대화 참여자 간의 감정 교환은 음성 운율(Prosody), 음향적 특징(Acoustic Features), 그리고 화자 간 감정 전이에 강력한 인과적 영향을 받는다. 본 연구에서는 텍스트 데이터를 배제하고 오직 음성신호만을 활용한다. 

본 연구는 "상대방 화자 B의 음성 감정 상태 및 발화 궤적이 화자 A의 다음 발화 감정 상태 전환에 어떤 인과적 영향을 미치는가?"라는 핵심 질문에서 출발한다. 

단순히 현재 발화의 감정을 분류하는 기존 SER(Speech Emotion Recognition)의 한계를 넘어, 이전 발화의 음성 히스토리와 상대방의 정서적 궤적을 통합하여 화자 A의 차기 감정 상태를 다차원 및 범주형으로 예측하는 End-to-End PipeLine을 제시한다.

본 연구는 대화 맥락 속에서 **화자 간 감정 전이** 및 상대방 감정이 나의 다음 감정에 미치는 인과적 영향을 분석합니다. 


## 2. Key Objectives & Hypothesis 
1. Acoustic Emotional causality Analysis: 화자 A의 음성 기반 감정 변화가 자발적인지, 혹은 화자 B의 음성 발화에 의한 감정 전이 현상에 기인한 지 인과적 관계를 분석한다.

* Dataset: IEMOCAP(Interactive Emotional Dyadic Motion Capture Database)

## 3. System Architecture & Methodology 
본 프레임워크는 음성 감정 인식과 대화 흐름 기반 감정 예측의 두가지 핵심 축으로 구성이 된다. 

## 4. Dataset & Preprocessing 
* Dataset: IEMOCAP(Interactive Emotional Dyadic motion Capture Database) - Audio Streams
* Data Processing:

## 5. Business & Service Insights 
* 고객센터 상담원 실시간 어시스트: 고객의 목소리 톤과 감정 전이 궤적을 추적하여 상담원의 화면에 고객의 예상 감정을 팝업으로 제공
  예) 경고: 목소리 톤 기반 불만 격화 위험 85%
* 음성 기반 전이 분석: 목소리 톤 변화가 일어나는 시점과 반응 속도를 파악하여 서비스 대응 가이드라인 마련


