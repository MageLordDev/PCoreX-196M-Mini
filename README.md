PCoreX-196M-Mini
Repository navigation
Code
Issues
Pull requests
PCoreX-196M-Mini
/README.md
MageLordDev
MageLordDev
1 minute ago
110 lines (89 loc) · 4.46 KB
Preview
Code
Blame
TR-EN 200M — Hugging Face + TPU Compatible Language Model
This model has not yet been trained; it is a pure, unweighted architecture. Anyone is free to adapt it to their own language or system, train it, and publish it. I would appreciate it if you would cite it.
A decoder-only (GPT style) language model with ~196M parameters for Turkish + English. Now a complete Hugging Face transformers architecture: PreTrainedModel based, registered with AutoModelForCausalLM, directly compatible with save_pretrained/push_to_hub/Trainer/accelerate/TPU (torch_xla). Weights are still randomly initialized — you will handle the training yourself.

Files
File Description
configuration_tr_en.py TrEnConfig(PretrainedConfig) — model hyperparameters
modeling_tr_en.py TrEnForCausalLM(PreTrainedModel, GenerationMixin) — architecture
tokenizer.py TR+EN byte-level BPE tokenizer training (raw tokenizer output)
convert_tokenizer.py Converts the raw tokenizer to the standard HF tokenizer folder
train.py Ready-made training framework with HF Trainer (TPU/GPU same code)
requirements.txt Required packages
Architecture (~196M parameters)
RoPE position encoding, RMSNorm, SwiGLU MLP, tied embeddings, KV-cache supported (model.generate() works), gradient checkpointing supported.

Settings Value
n_layer 10
n_embd 1152
n_head 16 (head_dim=72)
block_size 2048
vocab_size 32,000
Why compatible with TPU/HF?
PreTrainedModel legacy: save_pretrained, from_pretrained, push_to_hub, .half()/.to(dtype), device_map, PEFT/LoRA integration are automatically included.
AutoModelForCausalLM.register: Model can be loaded with trust_remote_code=True as AutoModelForCausalLM.from_pretrained("./tr_en_llm_ckpt").
Eager/SDPA attention: torch.nn.functional.scaled_dot_product_attention is used; works seamlessly with XLA graph compilation in TPU (no dependency on CUDA-specific kernels like flash-attn). Trainer + Accelerate: In TPU, distributed training, gradient accumulation, and mixed precision (BF16 — TPU's native format) are all automatically managed via TrainingArguments. Step-by-step usage
1) Train the Tokenizer
pip install -r requirements.txt
python tokenizer.py --files data_tr.txt data_en.txt --vocab_size 32000 --out tr_en_tokenizer.json --test
2) Convert the Tokenizer to HF format
python convert_tokenizer.py --tokenizer_json tr_en_tokenizer.json --out ./tr_en_llm_ckpt
3) Verify the architecture (optional, quick health check)
python modeling_tr_en.py
# -> number of parameters, random init loss value and save_pretrained test
4) Start training on TPU
pip install torch_xla # Wheel suitable for your TPU/Python version: https://github.com/pytorch/xla
accelerate config # "This machine" -> "TPU" -> number of cores
accelerate launch train.py \
--tokenizer_dir ./tr_en_llm_ckpt \
--train_file data_tr_en.txt \
--output_dir ./tr_en_llm_out \
--bf16 \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 4

The same command works on the GPU (select GPU in accelerate config).

5) Post-training loading
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("./tr_en_llm_out")
model = AutoModelForCausalLM.from_pretrained("./tr_en_llm_out", trust_remote_code=True)
out = model.generate(**tok("Merhaba,", return_tensors="pt"), max_new_tokens=30)
print(tok.decode(out[0]))
6) If you want to upload to Hub
model.push_to_hub("username/tr-en-gpt-200m")
tok.push_to_hub("username/tr-en-gpt-200m")
Important notes
train.py is a skeleton; Decisions such as dataset size, batch size, learning rate, total number of steps/epochs, and checkpoint frequency depend on your training plan — adjust them according to your own data.
It is recommended to use bf16=True in TPU (TPU's native precision format is more stable than fp16).
Torch_xla installation varies depending on your TPU hardware, PyTorch, and Python version; follow the official installation instructions (link above).
Since there is no network access and GPU/TPU in this environment, the code could not be run and trained here; only syntax validation (py_compile) was performed. I recommend testing the architecture with python modeling_tr_en.py in your own TPU environment.
