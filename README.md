# Causal checker

`causal checker` is a library developped to run causal analyses on LLM at scale. This repo is the code base for the project "A Universal Emergent Decomposition of  Retrieval Tasks in Autoregressive Language Models".

This repo contains
* `causal_checker/alignement.py`: A simple implementation of the causal abtraction framework to verify alignement between LM and high-level causal graph by running interchange intervention.
* `causal_checker/retrieval.py`: A definition of a high-level causal graph for retrieval tasks.
* `causal_checker/datasets`: 6 datasets sharing the same abstract input representation to study retrieval tasks.
* `demo/causal_checker_sweep.py` : a script using the `alignement` code to run residual stream patching on 13 models on all the datasets.
* `data_analysis`: folder with the data of residual-stream patching experiments, and code to reproduce the plots from "A Universal Emergent Decomposition of  Retrieval Tasks in Autoregressive Language Models".
* `internal_process_supervision` an application of request-patching to remove the effect of distractors on model solving a question-answering task.
* `mech_analysis` contains the code for a detailed case study on pythia-2.8 on the NanoQA dataset.


`demo/main_demo.py` walk you through the most important object of the code base.

##### Dependencies

This librairy is build on [swap-graphs](https://github.com/aVariengien/swap-graphs) for the objects representing model components and positions, and [TransformerLens](https://github.com/neelnanda-io/TransformerLens) for fine-grained hooks. For memory efficiency, HuggingFace hooks are also supported, but allow less control. 

