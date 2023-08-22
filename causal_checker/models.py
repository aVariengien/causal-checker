from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
import torch


def get_nb_layers(model):
    pass


def get_model(
    model_name, dtype=torch.bfloat16, cache_dir="/mnt/ssd-0/alex-dev/hf_models"
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
        cache_dir=cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
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
    assert size in ["70m", "160m", "410m", "1b", "2.8b", "6.9b", "12b"]
    model_name = f"EleutherAI/pythia-{size}"
    return get_model(model_name, dtype=dtype)


def get_model_and_tokenizer(model_name, dtype=torch.bfloat16):
    if "falcon" in model_name[:7]:
        return get_falcon_model(model_name.replace("falcon-", ""), dtype=dtype)
    elif "gpt2" in model_name:
        return get_gpt2_model(model_name.replace("gpt2-", ""), dtype=dtype)
    elif "pythia" in model_name:
        return get_pythia_model(model_name.replace("pythia-", ""), dtype=dtype)
    else:
        return get_model(model_name, dtype=dtype)
