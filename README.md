# TR-EN 200M — Hugging Face + TPU Uyumlu Dil Modeli

# Bu modele henüz her hangi bir eğtim yapılmamıştır model ağırlıksız saf mimaridir isteyen kendi diline veya sistemine uyarlayıp eğitip yayınlamakta özgürdür, atıf yaparsanız sevinirim

Türkçe + İngilizce için ~196M parametreli, decoder-only (GPT tarzı) bir dil modeli.
Artık tam bir **Hugging Face `transformers` mimarisi**: `PreTrainedModel` tabanlı,
`AutoModelForCausalLM` ile kayıtlı, `save_pretrained`/`push_to_hub`/`Trainer`/
`accelerate`/**TPU (torch_xla)** ile doğrudan uyumlu. Ağırlıklar hâlâ rastgele
başlatılmış — eğitimi siz üstleneceksiniz.

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `configuration_tr_en.py` | `TrEnConfig(PretrainedConfig)` — model hiperparametreleri |
| `modeling_tr_en.py` | `TrEnForCausalLM(PreTrainedModel, GenerationMixin)` — mimari |
| `tokenizer.py` | TR+EN byte-level BPE tokenizer eğitimi (ham `tokenizers` çıktısı) |
| `convert_tokenizer.py` | Ham tokenizer'ı standart HF tokenizer klasörüne çevirir |
| `train.py` | HF `Trainer` ile hazır eğitim iskeleti (TPU/GPU aynı kod) |
| `requirements.txt` | Gerekli paketler |

## Mimari (~196M parametre)

RoPE pozisyon kodlama, RMSNorm, SwiGLU MLP, tied embeddings, KV-cache destekli
(`model.generate()` çalışır), gradient checkpointing destekli.

| Ayar | Değer |
|---|---|
| n_layer | 10 |
| n_embd | 1152 |
| n_head | 16 (head_dim=72) |
| block_size | 2048 |
| vocab_size | 32.000 |

## Neden TPU/HF ile uyumlu?

- **`PreTrainedModel` mirası:** `save_pretrained`, `from_pretrained`, `push_to_hub`,
  `.half()/.to(dtype)`, `device_map`, PEFT/LoRA entegrasyonu otomatik gelir.
- **`AutoModelForCausalLM.register`:** Model, `trust_remote_code=True` ile
  `AutoModelForCausalLM.from_pretrained("./tr_en_llm_ckpt")` şeklinde yüklenebilir.
- **Eager/SDPA attention:** `torch.nn.functional.scaled_dot_product_attention`
  kullanılıyor; TPU'da XLA graph derlemesiyle sorunsuz çalışır (flash-attn gibi
  CUDA'ya özel kernellere bağımlılık yok).
- **`Trainer` + `accelerate`:** TPU'da dağıtık eğitim, gradient accumulation,
  mixed precision (bf16 — TPU'nun native formatı) hepsi `TrainingArguments`
  üzerinden otomatik yönetilir.

## Adım adım kullanım

### 1) Tokenizer'ı eğitin
```bash
pip install -r requirements.txt
python tokenizer.py --files data_tr.txt data_en.txt --vocab_size 32000 --out tr_en_tokenizer.json --test
```

### 2) Tokenizer'ı HF formatına çevirin
```bash
python convert_tokenizer.py --tokenizer_json tr_en_tokenizer.json --out ./tr_en_llm_ckpt
```

### 3) Mimariyi doğrulayın (opsiyonel, hızlı sağlık kontrolü)
```bash
python modeling_tr_en.py
# -> parametre sayısı, rastgele init loss değeri ve save_pretrained testi
```

### 4) TPU üzerinde eğitime başlayın
```bash
pip install torch_xla   # TPU/Python sürümünüze uygun wheel: https://github.com/pytorch/xla
accelerate config       # "This machine" -> "TPU" -> çekirdek sayısı
accelerate launch train.py \
    --tokenizer_dir ./tr_en_llm_ckpt \
    --train_file data_tr_en.txt \
    --output_dir ./tr_en_llm_out \
    --bf16 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4
```

GPU'da da **aynı komutla** çalışır (`accelerate config`'te GPU seçin).

### 5) Eğitim sonrası yükleme
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("./tr_en_llm_out")
model = AutoModelForCausalLM.from_pretrained("./tr_en_llm_out", trust_remote_code=True)

out = model.generate(**tok("Merhaba,", return_tensors="pt"), max_new_tokens=30)
print(tok.decode(out[0]))
```

### 6) Hub'a yüklemek isterseniz
```python
model.push_to_hub("kullanici-adi/tr-en-gpt-200m")
tok.push_to_hub("kullanici-adi/tr-en-gpt-200m")
```

## Önemli notlar

- `train.py` bir **iskelet**tir; veri seti boyutu, batch size, learning rate,
  toplam adım/epoch sayısı, checkpoint sıklığı gibi kararlar eğitim planınıza
  bağlı — kendi verinize göre düzenleyin.
- TPU'da `bf16=True` kullanmanız önerilir (TPU'nun native precision formatı,
  fp16'dan daha stabildir).
- `torch_xla` kurulumu TPU donanımına, PyTorch ve Python sürümünüze göre
  değişir; resmi kurulum talimatlarını takip edin (yukarıdaki link).
- Bu ortamda ağ erişimi ve GPU/TPU olmadığı için kodlar burada **çalıştırılıp
  eğitilemedi**; sadece sözdizimi doğrulandı (`py_compile`). Kendi TPU
  ortamınızda `python modeling_tr_en.py` ile mimariyi test etmenizi öneririm.
