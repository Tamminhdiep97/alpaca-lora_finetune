import torch
from peft import PeftModel
import transformers
import os, time
import tempfile

#assert ("LlamaTokenizer" in transformers._import_structure["models.llama"]), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

BASE_MODEL = "decapoda-research/llama-7b-hf"
LORA_WEIGHTS = "./lora-alpaca/checkpoint-205000/"

force_cpu = False

if torch.cuda.is_available() and not force_cpu:
    device = "cuda"
    print("Video memory available:", torch.cuda.get_device_properties(0).total_memory / 1024 / 1024, "MBs")
else:
    device = "cpu"
print("Compute device is:", device)

print("Loading model with selected weights ...")

if device == "cuda":

    print("model on cuda")

    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="offload",
    )

    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="offload",

    )

model = model.merge_and_unload()
model.save_pretrained("/merge/ja_en_alpaca_lora_7b")



