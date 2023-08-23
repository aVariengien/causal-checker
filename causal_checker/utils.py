# %%

from attrs import define, field
from typing import List, Callable, Dict, Tuple, Set, Optional, Any, Literal
import torch
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from hashlib import sha1


def get_first_token_id(tokenizer: Any, text: str) -> int:
    if tokenizer is None:
        sha1_hash = sha1(text.encode()).hexdigest()
        return int(sha1_hash, 16)
    idx_first_tok = 0
    if isinstance(tokenizer, LlamaTokenizerFast):
        if text[0] == " ":
            text = text[
                1:
            ]  # special case because of LLAMA sentence piece tokenizer ignoring the first space
        idx_first_tok = 1  # ignore automatic bos
        if text[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            idx_first_tok = 2  # ignore automatic space for the digits special cases
        if "<prefix>" in text and "</prefix>" in text:
            text = text.replace("<prefix>", "").replace("</prefix>", "")
            idx_first_tok = 2  # prefix contains a previous token influencing the tokenization of the next

    first_token_id = torch.tensor(tokenizer([text])["input_ids"])[0][idx_first_tok]
    return int(first_token_id)


def get_first_token(tokenizer: Any, text: str) -> str:
    if tokenizer is None:
        return text
    first_token_id = get_first_token_id(tokenizer, text)
    tok = tokenizer.decode(first_token_id)
    return tok


# %%
