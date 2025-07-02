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
   상대적으로 적은 파라미터로도 로컬에서 빠르고 안정적인 추론 가능

> 📌 **활용 예시**: 뉴스, 칼럼, 회의록 등 실용 텍스트를 논리적이며 압축력 있게 재구성



## 🏗️ 모델 구조 및 아키텍처

입력 문장 + 프롬프트
↓
[ KoBART Seq2Seq 모델 ]
↓
출력 문장 (풍부하고 자연스러운 요약/이어쓰기 결과)

python
복사
편집

- **인코더**: 입력 문장을 벡터로 인코딩  
- **디코더**: 문맥 기반으로 이어지는 문장 생성  



## 🧪 파인튜닝 코드 예시

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments

model_name = "EbanLee/kobart-summary-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

dataset = load_dataset("json", data_files={"train": "path/to/train_summarized_1000.jsonl"})

prompt = (
    "다음 기사 내용을 단순하게 요약하지 말고, 서사 구조와 맥락을 살려 3~5문장으로 풍부하게 요약해줘. "
    "이슈의 배경과 원인, 참여자들의 주장, 사회적 의미와 쟁점을 포함해줘. 객관적인 어조로 서술해줘.\n\n"
)

def preprocess_function(examples):
    inputs = [prompt + "문장: " + text + "\n\n글:" for text in examples["input"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
    labels = tokenizer(examples["output"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset["train"].map(preprocess_function, batched=True, remove_columns=["input", "output"])

training_args = TrainingArguments(
    output_dir="./kobart_prompt_tuned2",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=3e-2,
    logging_steps=100,
    save_steps=500,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()
🚀 추론 코드 예시
python
복사
편집
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "hienchong/Maildan_kobart_v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

prompt = (
    "다음 기사 내용을 단순하게 요약하지 말고, 서사 구조와 맥락을 살려 3~5문장으로 풍부하게 요약해줘. "
    "이슈의 배경, 원인, 주장 내용, 사회적 의미를 포함해줘. 감정적이지 않고 객관적으로 서술해줘.\n\n"
)

text = (
    "문화체육관광부는 경복궁 복원 사업의 일환으로 광화문 현판을 한글로 교체하자는 논의를 시작했다. "
    "해당 제안은 유인촌 장관의 요청에서 비롯되었으며, 현재까지는 한자로 된 현판이 사용되고 있다..."
)

input_text = prompt + "문장: " + text + "\n\n글:"
inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).to(model.device)

summary_ids = model.generate(**inputs, max_new_tokens=200, num_beams=4)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(summary)
```

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
