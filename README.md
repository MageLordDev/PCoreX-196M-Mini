# TR-EN 200M — Hugging Face + TPU Compatible Language Model

A ~196M-parameter, decoder-only (GPT-style) language model for Turkish + English.
It is a full **Hugging Face `transformers` architecture**: built on `PreTrainedModel`,
registered for `AutoModelForCausalLM`, and directly compatible with
`save_pretrained` / `push_to_hub` / `Trainer` / `accelerate` / **TPU (torch_xla)**.
The weights are randomly initialized (an "empty" model) — training is up to you.

> This is the 196M variant. The same code scales to 1B by changing only the config
> values; see [`README.md`](README.md) for the 1B model.

## Files

| File | Description |
|---|---|
| `configuration_tr_en.py` | `TrEnConfig(PretrainedConfig)` — model hyperparameters |
| `modeling_tr_en.py` | `TrEnForCausalLM(PreTrainedModel, GenerationMixin)` — the architecture |
| `tokenizer.py` | Trains the TR+EN byte-level BPE tokenizer (raw `tokenizers` output) |
| `convert_tokenizer.py` | Converts the raw tokenizer into a standard HF tokenizer folder |
| `train.py` | Ready-made training skeleton built on HF `Trainer` (same code on TPU/GPU) |
| `requirements-gpu.txt` / `requirements-tpu.txt` | Version-pinned dependencies per environment |

## Architecture (~196M parameters)

RoPE positional encoding, RMSNorm, SwiGLU MLP, tied embeddings, KV-cache support
(`model.generate()` works), gradient checkpointing support.

| Setting | Value |
|---|---|
| n_layer | 10 |
| n_embd | 1152 |
| n_head | 16 (head_dim = 72) |
| ffn hidden (SwiGLU) | 3072 |
| block_size | 2048 |
| vocab_size | 32,000 |

Parameter count:
```
total = vocab*n_embd + n_layer * (4*n_embd² + 3*n_embd*ffn_hidden)
      = 32000*1152 + 10 * (4*1152² + 3*1152*3072)
      = 196,116,480   (196,140,672 including RMSNorm weights, ~196.1M)
```

### Using the 196M configuration

The default values in `configuration_tr_en.py` are set for the 1B model. To build the
196M variant, pass the values explicitly:

```python
from configuration_tr_en import TrEnConfig
from modeling_tr_en import TrEnForCausalLM

config = TrEnConfig(vocab_size=32000, n_layer=10, n_head=16, n_embd=1152, block_size=2048)
model = TrEnForCausalLM(config)
print(f"{model.num_params() / 1e6:.1f}M parameters")
```

Train the tokenizer with a matching vocabulary size:
`python tokenizer.py ... --vocab_size 32000`.

## Why it works with TPU / Hugging Face

- **`PreTrainedModel` inheritance:** `save_pretrained`, `from_pretrained`, `push_to_hub`,
  `.half()` / `.to(dtype)`, `device_map`, and PEFT/LoRA integration come for free.
- **Auto class registration:** `register_for_auto_class` is used, so the model can be
  loaded with `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`.
- **SDPA attention:** uses `torch.nn.functional.scaled_dot_product_attention`, which
  compiles fine under XLA on TPU (no dependency on CUDA-only kernels such as flash-attn).
- **`Trainer` + `accelerate`:** distributed TPU training, gradient accumulation and
  mixed precision (bf16, the TPU-native format) are all handled through
  `TrainingArguments`.

## Step-by-step usage

### 1) Train the tokenizer
```bash
pip install -r requirements-gpu.txt     # or requirements-tpu.txt on TPU
python tokenizer.py --files data_tr.txt data_en.txt --vocab_size 32000 --out tr_en_tokenizer.json --test
```

### 2) Convert the tokenizer to HF format
```bash
python convert_tokenizer.py --tokenizer_json tr_en_tokenizer.json --out ./tr_en_llm_ckpt
```

### 3) Sanity-check the architecture (optional)
```bash
python modeling_tr_en.py
# prints parameter count, the random-init loss and runs a save_pretrained test
```
Note that the script builds the default (1B) config; use the snippet above to check
the 196M variant.

### 4) Train on TPU
```bash
accelerate config       # "This machine" -> "TPU" -> number of cores
accelerate launch train.py \
    --tokenizer_dir ./tr_en_llm_ckpt \
    --train_file data_tr_en.txt \
    --output_dir ./tr_en_llm_out \
    --bf16 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4
```

The same command works on GPU (choose GPU in `accelerate config`).

### 5) Load after training
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("./tr_en_llm_out")
model = AutoModelForCausalLM.from_pretrained("./tr_en_llm_out", trust_remote_code=True)

out = model.generate(**tok("Merhaba,", return_tensors="pt"), max_new_tokens=30)
print(tok.decode(out[0]))
```

### 6) Push to the Hub (optional)
```python
model.push_to_hub("your-username/tr-en-gpt-200m")
tok.push_to_hub("your-username/tr-en-gpt-200m")
```

## Tokenizer design (Turkish + English)

- **Algorithm:** byte-level BPE (GPT-2/GPT-4 style). It splits Turkish's agglutinative
  morphology into useful subwords and never hits unknown characters (no OOV).
- **Normalization:** NFC, so Turkish-specific characters (İ, ı, ş, ğ, ç, ö, ü) stay
  consistent and intact.
- **No lowercasing:** the distinction between "İstanbul" / "istanbul" and İ/i, I/ı is
  meaningful in Turkish, so case is preserved.
- **Special tokens:** `<pad> <unk> <bos> <eos> <sep> <mask>` plus
  `<|user|> <|assistant|> <|system|>` for chat formatting.
- **Vocabulary size:** 32,000 for the 196M model.

## Notes

- `train.py` is a **skeleton**: dataset size, batch size, learning rate, total
  steps/epochs and checkpoint frequency depend on your training plan.
- On TPU, `bf16` is recommended (the native precision format, more stable than fp16).
- `torch_xla` must match your `torch` version exactly; see `requirements-tpu.txt`.
- The code has only been syntax-checked (`py_compile`); it has not been run or
  trained in the environment it was written in. Run `python modeling_tr_en.py` on your
  own setup first.
