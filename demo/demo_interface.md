# Demo Interface 
## 1. Overview 
본 시연 화면은 Friends 영상과 발화별 감정 분석 결과를 실시간으로 연결하여 대화 속 감정 변화 과정을 시각적으로 보여주는 것을 목표로 한다. 

영상 재생 시점에 맞춰 현재 화자, 발화 내용, VAD 감정 점수 및 감정 변화 그래프를 표시 

## 2. Screen Layout 
시연 화면은 다음 네 가지 영역으로 구성한다. 

### 2.1 Video Player 
* 프렌즈 영상 재생
* 재생 및 일시정지
* 영상 구간 이동
* 현재 영상 재생 시간 표시

### 2.2 Current Utterance 
* 현재 발화자의 이름 또는 ID
* 발화 시작 및 종료 시간
* 현재 발화 내용

### 2.3 Emotion Score 
현재 발화의 감정 분석 결과를 표시한다. 
* valence
* Arousal
* Dominance

각 점수는 해당 발화의 모델 예측 결과를 사용한다. 

### 2.4 Emotion Change Graph 
대화가 진행됨에 따라 화자별 감정 점수가 어떻게 변화하는지 그래프로 나타낸다. 

* X-axis: 발화 순서 또는 영상 재생 시간
* Y-axis: VAD 점수
* Series: 화자별 감정 변화
* Current Position: 현재 영상 재생 위치

Valence, Arousal, Dominance는 각각 구분하여 확인 할 수 있도록 구성하기 

## 3. Synchronization 
영상과 감정 분석 결과는 발화별 시작 시간 및 종료 시간을 기준으로 연결 

1. 영상의 현재 재생 시간을 확인한다.
2. 해당 시간에 대응하는 발화 데이터를 찾는다.
3. 현재 화자와 발화 내용을 표시한다.
4. 해당 발화의 VAD 점수를 표시한다.
5. 그래프에 현재 재생 위치를 표시한다.

영상의 재생, 일시정지 및 구간 이동 상황에서도 정보가 올바르게 표시되는지 확인하기 

## 4. Main Features

| Feature | Description |
|---|---|
| Video Playback | 영상 재생 및 일시정지 |
| Time Navigation | 원하는 영상 구간으로 이동 |
| Speaker Display | 현재 발화자 표시 |
| Utterance Display | 현재 발화 내용 표시 |
| VAD Score Display | 발화별 감정 점수 표시 |
| Emotion Graph | 화자별 감정 변화 시각화 |
| Timeline Synchronization | 영상 시간과 분석 결과 연결 |

## 5. Implementation Priority

### Priority 1: Essential Features
- 영상 재생 및 일시정지
- 현재 발화 정보 표시
- 발화별 VAD 점수 표시
- 화자별 감정 변화 그래프

### Priority 2: Additional Features
- 영상 구간 이동
- 최종 시연 장면 선택 기능
- VAD 차원별 그래프 전환

## 6. Testing

시연 전 다음 항목을 확인한다.

- 영상 재생 시간과 발화 정보가 일치하는가?
- 현재 화자 정보가 정확하게 표시되는가?
- 발화별 감정 점수가 올바르게 연결되는가?
- 영상 재생에 따라 그래프의 현재 위치가 이동하는가?
- 일시정지 후에도 정보가 유지되는가?
- 구간 이동 시 해당 시점의 발화와 감정 점수가 표시되는가?

*Note: 실제 구현 기능과 화면 구성은 개발 진행 상황에 따라 업데이트한다.*

