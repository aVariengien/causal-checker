from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
import torch


def get_model(model_name, dtype=torch.bfloat16):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_falcon_model(size, dtype=torch.bfloat16):
    assert size in ["7b-instruct", "7b", "40b", "40b-instruct"]
    model_name = f"/mnt/ssd-0/falcon-{size}"
    return get_model(model_name, dtype=dtype)


def get_gpt2_model(size, dtype=torch.bfloat16):
    assert size in ["small", "medium", "large", "xl"]
    if size == "small":
        model_name = "gpt2"
    else:
        model_name = f"gpt2-{size}"
    return get_model(model_name, dtype=dtype)


def get_pythia_model(size, dtype=torch.bfloat16):
    assert size in ["70m", "125m", "350m", "2.8b", "6.9b", "12b"]
    model_name = f"EleutherAI/pythia-{size}"
    return get_model(model_name, dtype=dtype)
