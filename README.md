# Causal checker

`causal checker` is a library developped to run causal analyses on LLM at scale. This repository is the code base for the project "A Universal Emergent Decomposition of  Retrieval Tasks in Autoregressive Language Models". The code is in still in early developement stage.

The repo contains
* `causal_checker/alignement.py`: A simple implementation of the [causal abstraction framework](https://arxiv.org/abs/2106.02997) to verify alignement between LM and high-level causal graph by running interchange interventions.
* `causal_checker/retrieval.py`: A definition of a high-level causal graph for retrieval tasks.
* `causal_checker/datasets`: 6 datasets sharing the same abstract input representation to study retrieval tasks.
* `demo/causal_checker_sweep.py` : a script using to run residual stream patching on 13 models on all datasets.
* `data_analysis`: Data from residual-stream patching experiments, and code to reproduce the plots from "A Universal Emergent Decomposition of  Retrieval Tasks in Autoregressive Language Models".
* `internal_process_supervision`: An application of request-patching to remove the effect of distractors on model solving a question-answering task.
* `mech_analysis`: The code for a detailed case study on pythia-2.8 on the NanoQA dataset.

##### To start

`demo/main_demo.py` walk you through the most important object of the code base.

##### Dependencies

This librairy is build on [swap-graphs](https://github.com/aVariengien/swap-graphs) for the objects representing model components and positions, and [TransformerLens](https://github.com/neelnanda-io/TransformerLens) for fine-grained hooks. For memory efficiency, HuggingFace hooks are also supported, but allow less control. 

##### Install

`pip install -e .`