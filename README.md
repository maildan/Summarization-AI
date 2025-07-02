# 🤖 Maildan_AI

모델 페이지: https://huggingface.co/hienchong/Maildan_kobart_v3

## 📘 Maildan_kobart_v3 모델 카드

`Maildan_kobart_v3`는 한국어 문서의 **서사적 요약** 및 **자연스러운 이어쓰기**를 목표로 파인튜닝된 KoBART 기반 Transformer 모델입니다.  
프롬프트 튜닝 기반의 학습 방식을 활용해 문맥을 이해하고 풍부한 요약문을 생성하는 데 강점을 보입니다.



## 🧠 모델 개요

| 항목 | 설명 |
|------|------|
| **모델 타입** | Transformer 기반 Seq2Seq (KoBART) |
| **학습 목적** | 뉴스, 기사, 회의록, 이메일 등의 한국어 문서를 요약하거나 이어쓰기 |
| **튜닝 방식** | 프롬프트 기반 파인튜닝 (Prompt + 문맥 입력 → 이어쓰기/요약 출력) |
| **기반 모델** | [`EbanLee/kobart-summary-v3`](https://huggingface.co/EbanLee/kobart-summary-v3) |
| **출력 형태** | 배경, 원인, 쟁점 등을 반영한 자연스러운 서사형 요약 문장 |



## ✨ 주요 기능

1. **서사적 요약 생성**  
   단순 요약이 아닌, 논리 구조와 의미 흐름을 담은 3~5문장 요약 생성

2. **프롬프트 기반 제어**  
   지시문(prompt)을 통해 문체, 정보 밀도, 어조를 유연하게 제어 가능

3. **경량 추론 최적화**  
   상대적으로트

## 🛠️ 설치 모듈 및 역할

```python
bash
pip install torch transformers accelerate datasets peft
```

| 모듈                 | 설명                                                  |
| ------------------ | --------------------------------------------------- |
| **`torch`**        | PyTorch 딥러닝 프레임워크로, 모델 학습/추론의 핵심 역할 수행              |
| **`transformers`** | Hugging Face에서 제공하는 모델, 토크나이저, `Trainer` 등 핵심 도구 모음 |
| **`accelerate`**   | 다양한 환경(CPU, GPU, TPU 등)에서 손쉬운 분산 학습 및 실행 지원         |
| **`datasets`**     | JSONL 포함 다양한 형식의 데이터셋 로딩 및 전처리 기능 제공                |
| **`peft`**         | PEFT(프롬프트 튜닝, LoRA 등)를 통해 경량화된 파인튜닝 구현 가능           |


```python
bash
pip install peft
```
-> 프롬프트시 해당 모듈 설치치

## 📄 라이선스

- 기반 모델: Apache 2.0 License
- 학습 데이터: 비공개 (비상업적 연구 목적 사용 가능)

## 🙋‍♀️ 제작자 정보

- 이름: 류현정 (Hienchong)
- Hugging Face: https://huggingface.co/hienchong
