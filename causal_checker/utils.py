from attrs import define, field
from typing import List, Callable, Dict, Tuple, Set, Optional, Any, Literal
import torch



def get_first_token(tokenizer: Any, text: str) -> str:
    if tokenizer is None:
        return text
    first_token_id = torch.tensor(tokenizer([text])["input_ids"])[0][0]
    return tokenizer.decode(first_token_id)
