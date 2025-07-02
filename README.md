# AI
maildan_AI

## Maildan\_kobart\_v3 모델 카드

**Maildan\_kobart\_v3**는 한국어 텍스트 요약 및 자연스러운 이어쓰기 생성을 목적으로 개발된 모델입니다. 본 모델은 한국어에 특화된 사전학습 모델 **KoBART**(`gogamza/kobart-base-v2`)를 기반으로, 프롬프트 기반 seq2seq 방식으로 파인튜닝 되었습니다.

---

## 🧠 모델 개요

* **모델 타입**: Transformer 기반 Encoder-Decoder (KoBART)
* **사용 목적**: 기사, 뉴스, 회의록, 이메일 등의 한국어 문서를 요약하거나 서사적으로 이어쓰기
* **파인튜닝 방식**: 프롬프트 기반 학습 (Prompt + 문맥 입력 → 이어쓰기/요약 출력)
* **기반 모델**: [gogamza/kobart-base-v2](https://huggingface.co/gogamza/kobart-base-v2)
* **출력 형태**: 배경, 원인, 쟁점 등 문맥을 반영한 자연스러운 요약 또는 이어쓰기 문장

---

## 🔍 모델 주요 기능 (Top 3 Features)

1. **서사적이고 논리적인 요약 생성**: 단순 축약이 아닌, 배경-원인-결과의 흐름을 담은 맥락 중심 요약 생성
2. **프롬프트 기반 맞춤 응답**: 입력된 지시문(prompt)에 따라 다양한 스타일, 어조, 정보 밀도 반영 가능
3. **경량 구조로 빠른 추론**: 상대적으로 작은 파라미터 수로 로컬 환경에서도 효율적인 추론 성능 제공

> 📌 사용 범위: 한국어 뉴스, 이메일, 회의록 등의 문서를 논리적이고 풍부한 요약으로 재구성하는 데 활용됩니다.

---

## 🏗️ 모델 구조 및 아키텍처

```
입력 문장 + 프롬프트
        │
        ▼
──────────────────────────────
|       KoBART 모델 구조       |
|   - 인코더: 문장 벡터화       |
|   - 디코더: 이어쓰기/요약 생성 |
──────────────────────────────
        │
        ▼
 출력 문장 (풍부하고 자연스러운 요약문)
```

* **KoBART**는 한국어 BART 구조로, 문맥을 양방향으로 이해하고 자연스러운 출력을 생성하는 데 강점을 가집니다.
* Seq2Seq 구조 덕분에, 문장을 조건부로 생성할 수 있어 요약, 번역, 이어쓰기에 최적입니다.

---

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
    "이슈가 발생한 배경과 원인, 참여자들이 주장하는 핵심 내용이 주장에 담긴 사회적 의미나 쟁점이 포함되도록 작성해줘. "
    "감정적이지 않고 객관적인 어조를 유지하되, 논쟁의 구조는 드러나게 써줘.\n\n"
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
```

---

## 📌 추론 코드 예시

```python
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

---

## 🔓 라이선스

* 기반 모델: Apache 2.0 License
* 학습 데이터: 비공개 / 연구 및 비상업적 사용 가능

---

## 🧭 향후 계획

* ROUGE, BLEU 기반 정량 성능 수치 추가 예정
* Gradio/Streamlit 기반 데모 페이지 개발 예정
* 다양한 Task (메일 응답 생성, 뉴스 해설 등)로 세분화된 응용 확대

---

## 🙋‍♀️ 제작자 정보

* 이름: 류현정 (Hienchong)
* Hugging Face: [https://huggingface.co/hienchong](https://huggingface.co/hienchong)
* 모델 페이지: [https://huggingface.co/hienchong/Maildan\_kobart\_v3](https://huggingface.co/hienchong/Maildan_kobart_v3)
