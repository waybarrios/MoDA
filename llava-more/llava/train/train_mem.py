import sys
import os
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))
from llava.train.train import train

import os
import torch

from huggingface_hub import login

# Token is now read from environment variable



print("VISIBLE DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("VISIBLE TO TORCH:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"cuda:{i} → {torch.cuda.get_device_name(i)}")


def is_debug():
    return int(os.environ.get('DEBUG', 0))

if __name__ == "__main__":
    from huggingface_hub import login, whoami
    # Read token from environment (HF_TOKEN is preferred, HUGGINGFACE_HUB_TOKEN is legacy)
    TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if TOKEN:
        login(token=TOKEN, add_to_git_credential=False)  # no prompts, rank-safe
        print("HF whoami:", whoami())                    # sanity check
    else:
        print("WARNING: No HuggingFace token found in environment variables (HF_TOKEN or HUGGINGFACE_HUB_TOKEN)")

    
    if is_debug():
        train(attn_implementation=None)
    else:
         
        train(attn_implementation="flash_attention_2")
