from .experiments import *
from .plot_fn import *

# %%
import math
import random as rd
from functools import partial
from pprint import pprint
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch

from swap_graphs.core import WildPosition
from swap_graphs.datasets.nano_qa.nano_qa_dataset import (
    NanoQADataset,
    evaluate_model,
    pprint_nanoqa_prompt,
)
from swap_graphs.datasets.nano_qa.nano_qa_utils import (
    check_tokenizer_nanoQA,
    print_performance_table,
)
from swap_graphs.utils import clean_gpu_mem, printw
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint  # Hooking utilities

from experiments import *  # type: ignore
from plot_fn import *  # type: ignore
from transformer_lens.utils import test_prompt

torch.set_grad_enabled(False)

# %%

model_name = "pythia-2.8b"
model = HookedTransformer.from_pretrained(
    model_name=model_name, cache_dir="/mnt/ssd-0/alex-dev/hf_models"
)

MID_LAYER = None
if model_name == "pythia-2.8b":
    MID_LAYER = 16
else:
    raise ValueError(f"no mid layer set for {model_name}")

# %% playing with tensors

idx = torch.randint(0, 30, (10, 10, 30))
x = torch.rand((10, 10, 30))


b = x[
    torch.arange(x.size(0)).unsqueeze(1).unsqueeze(2),
    torch.arange(x.size(1)).unsqueeze(0).unsqueeze(2),
    idx,
]

for i in range(10):
    for j in range(10):
        for k in range(30):
            assert x[i, j, idx[i, j, k]] == b[i, j, k]

# %%
nano_qa_dataset = NanoQADataset(
    name="general",
    nb_samples=2,
    nb_variable_values=5,
    querried_variables=ALL_NAR_VAR,
    tokenizer=model.tokenizer,  # type: ignore
    seed=None,
)
# %%
cache = {}
fill_cache(cache, model, nano_qa_dataset, mid_layer=MID_LAYER)
# %%
logits = model(nano_qa_dataset.prompts_tok)
end_position = WildPosition(position=nano_qa_dataset.word_idx["END"], label="END")
logit_end = logits[
    range(len(nano_qa_dataset)),
    end_position.positions_from_idx(list(range(len(nano_qa_dataset)))),
    :,
]


def model_end(x):
    res = model.ln_final(x.cuda().unsqueeze(0))
    return model.unembed(res).cpu().squeeze(0)


lg_end_test = model_end(cache["blocks.31.hook_resid_post-out-END"])
print(torch.linalg.norm(lg_end_test.cpu() - logit_end.cpu()))
# %% new tests
all_indices = []
for v in ALL_NAR_VAR:
    all_indices.append(
        torch.tensor(nano_qa_dataset.narrative_variables_token_id[v]).unsqueeze(-1)
    )

idx = torch.cat(all_indices, dim=-1)
# %%

a = torch.rand((12,12,len(nano_qa_dataset), 50000))

# %%
# Extend idx to have the same number of dimensions as a
expanded_idx = idx.expand((*a.shape[:-2], *idx.shape))


expanded_idx2 = idx.unsqueeze(0).expand((*a.shape[:-2], *idx.shape))

assert torch.all(expanded_idx == expanded_idx2)

# %%
# Create a new tensor b by indexing a with the extended idx
b = a.gather(dim=-1, index=expanded_idx)
for i in range(len(nano_qa_dataset)):
    for j in range(len(ALL_NAR_VAR)):
        assert torch.all(a[..., i, idx[i, j]] == b[..., i, j])

# %%


# %%
model.reset_hooks()
res = model.ln_final(cache["blocks.31.hook_resid_post"])
# %%
torch.linalg.norm(model.unembed(res) - logits)

# %%
plt.hist(cache["ln_final.hook_scale-out-END"].flatten(), bins=100)


# %%
flat_ln = list(cache["ln_final.hook_scale-out-END"].flatten().numpy())
all_ratio = []
for x in range(100000):
    x1 = rd.choice(flat_ln)
    x2 = rd.choice(flat_ln)
    all_ratio.append(x1 / x2)
plt.hist(all_ratio, bins=100)
# %%
print(np.array(all_ratio).mean(), np.array(all_ratio).std())

# %%
