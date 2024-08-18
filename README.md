# FineTuning-RoBERTa-LLaMA-MultiNLI
This repository contains the implementation and analysis of various fine-tuning techniques applied to RoBERTa and LLaMA models on the MultiNLI dataset. The project is part of the fourth computer assignment for the Natural Language Processing course at the University of Tehran.

## Table of Contents

- [Introduction](#introduction)
- [Assignment Overview](#assignment-overview)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Results and Analysis](#results-and-analysis)
- [Report](#report)
- [License](#license)


## Introduction

This project explores traditional and advanced fine-tuning methods, including LoRA, P-tuning, and prompt-based techniques, applied to large language models like RoBERTa and LLaMA. The goal is to compare the effectiveness of these methods on the MultiNLI dataset.

## Assignment Overview

### Part 1: Research and Learning

- **Traditional Fine-Tuning Methods:** Research and learn about fine-tuning all parameters of a model versus fine-tuning specific layers.
- **LoRA Technique:** Investigate the LoRA (Low-Rank Adaptation) technique for fine-tuning models.
- **Prompting Methods:** Explore hard-prompt and soft-prompt methods.

### Part 2: Practical Implementation

1. **Fine-Tuning All Parameters:** Fine-tune all parameters of the `roberta-large` model.
2. **LoRA Fine-Tuning:** Fine-tune the `roberta-large` model using the LoRA method.
3. **P-Tuning:** Implement P-tuning on the `roberta-large` model.
4. **Zero and One Shot Prompting:** Apply zero-shot and one-shot prompting techniques using `llama-3-8b-instruct`.
5. **QLoRA Fine-Tuning:** Fine-tune `llama-3-8b-instruct` using the QLoRA technique.
6. **Classifier Layer with QLoRA:** Add a classifier layer on top of `llama-3-8b-instruct` and fine-tune it using the QLoRA technique.

## Dataset

We have used the MultiNLI dataset from Hugging Face. The dataset can be downloaded from the following link:

- **MultiNLI Dataset:** [MultiNLI on Hugging Face](https://huggingface.co/datasets/nyu-mll/multi_nli)

## Prerequisites

Before you begin, ensure you have the following requirements:

- Libraries: `transformers`, `datasets`, `torch`, `sklearn`, `numpy`, `peft`, `bitsandbytes`, `accelerate`, `pandas`
- Basic understanding of NLP, fine-tuning techniques, and prompt engineering.


## Installation

To clone and run this repository locally:
```sh
git clone https://github.com/PouyaGohari/FineTuning-RoBERTa-LLaMA-MultiNLI.git
cd FineTuning-RoBERTa-LLaMA-MultiNLI
```

## Usage
This project is organized into multiple Jupyter notebooks, each corresponding to different fine-tuning and prompt-based techniques:

- **Fine-Tuning RoBERTa:** [`Codes/CA4_Roberta_All.ipynb`](Codes/CA4_Roberta_All.ipynb)
- **LoRA Fine-Tuning:** [`Codes/CA4-Roberta-LORA.ipynb`](Codes/CA4-Roberta-LORA.ipynb)
- **P-Tuning RoBERTa:** [`Codes/roberta-p-tuning.ipynb`](Codes/roberta-p-tuning.ipynb)
- **Zero and One Shot Prompting:** [`Codes/lama-icl.ipynb`](Codes/lama-icl.ipynb)
- **QLoRA Fine-Tuning LLaMA:** [`Codes/qlora-frist-part .ipynb`](Codes/qlora-frist-part%20.ipynb)
- **Classifier Layer with QLoRA:** [`Codes/qlora-clf-lama.ipynb`](Codes/qlora-clf-lama.ipynb)

To run any of the models:

1. Open the corresponding notebook.
2. Follow the instructions to preprocess the data, train the model, and evaluate its performance.

Each notebook is self-contained, with detailed instructions and code explanations for each fine-tuning method and prompt technique.

## Results and Analysis

- **Traditional Fine-Tuning vs. LoRA (RoBERTa-Large):** Comparison of performance when fine-tuning all parameters vs. using the LoRA method on the `roberta-large` model.
- **P-Tuning Performance (RoBERTa-Large):** Analysis of how P-tuning impacts the performance of the `roberta-large` model.
- **Prompting Techniques (LLaMA):** Results from zero and one-shot prompting on the `llama-3-8b-instruct` model.
- **QLoRA with LLaMA:** Insights on using QLoRA for fine-tuning the `llama-3-8b-instruct` model.
- **Classifier Layer with QLoRA (LLaMA):** Evaluation of adding a classifier layer on top of `llama-3-8b-instruct` and fine-tuning it with the QLoRA technique.


## Report

A comprehensive report detailing the research, methodology, implementation, results, and analysis for each part of the assignment is available [here](Report/nlp_report.pdf).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

