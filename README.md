# Masterthesis_VQA
This repository contains all code files of my master's thesis "A Benchmark for Scientific Figure and Table Understanding with Multimodal LLMs", written at the University of Mannheim.

This benchmark tests the ability of Multimodal LLMs to reason about scientific figures and tables by applying a Visual Question-Answering (VQA) task. Therefore, the models are prompted questions, given an image and its caption. The model responses are stored in a separate csv file und later evaluated by diverse metrics. This benchmark includes traditional metrics from the field of machine translation (e.g., BLEU, ROUGE, etc.), but proposes also two new LLM-based metrics: LLM-Accuracy and VQA-MQM.

The tables and figures were extracted from scientific articles, which were published on arXiv. Only papers were included, which follow a Creative Commons license. The corresponding Question-Answer (QA) pairs were generated using the [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) model. The exact notebooks and scripts, which were used to create the datasets, can be found in the folder `dataset`. Please note that the paths in those code files are not necessarily up-to-date in case you want to run them.

## Model list

The following models are integrated into this benchmark. This list includes a shorter model name (which is required to run the inference script) and the exact version.

- GLM: [glm-4v-9b](https://huggingface.co/THUDM/glm-4v-9b)
- GPT4: [GPT-4o](https://openai.com/index/hello-gpt-4o/)
- InstructBlip: [instructblip-flan-t5-xxl](https://huggingface.co/Salesforce/instructblip-flan-t5-xxl)
- InternVL: [InternVL2_5-8B](https://huggingface.co/OpenGVLab/InternVL2_5-8B)
- LLaVA: [llava-v1.6-vicuna-13b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf)
- MiniCPM: [MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6)
- Ovis: [Ovis2-8B](https://huggingface.co/AIDC-AI/Ovis2-8B)
- Paligemma: [paligemma2-10b-mix-448](https://huggingface.co/google/paligemma2-10b-mix-448)
- Qwen: [Qwen2.5-VL-7B-Instruct](Qwen/Qwen2.5-VL-7B-Instruct)

## Installation

1.) **Clone this repository.**

2.) **Install the requirements:** `pip install -r requirements.txt`\
**Note:** The requirements are general. For specific models, check the Hugging Face model page to ensure the correct package versions are installed.

3.) **Download the following datasets:**
  - [SciFigData](https://huggingface.co/datasets/fuubian/SciFigData)
  - [SciTabData](https://huggingface.co/datasets/fuubian/SciTabData)
  - [VisualSciQA](https://huggingface.co/datasets/fuubian/VisualSciQA)

4.) **Ensure that the directory structure is correct.**
It should look like this:
```
.
└── data/
    ├── task1_test_split.csv
    ├── task1_train_split.csv
    ├── task2.csv
    ├── task3.csv
    ├── tables.csv
    ├── figures.csv
    ├── tables/
    │   ├── code/
    │   └── image/
    └── figures/
```

5.) **Set up API keys:**
  - For GPT-4:
    - Copy `env.example` to `.env`.
    - Add your OpenAI API key to the `.env` file.
  - For Paligemma:
    - Request access to the model on the Hugging Face model page.
    - Add your Hugging Face access token to the `.env` file.

    

## Running inference

To run a model on the QA dataset, execute the `inference.py` script with the following arguments:

```
python inference.py <task_number> <model_name> <use_table_code>
```

  -	`task_number`: The task number (1, 2, or 3).
  -	`model_name`: The exact model name (refer to the `config` file for available options).
  -	`use_table_code` Whether to use table code as input for answering questions (true or false). This is only applicable for Task 1.

The model responses will be written into an output directory.



## Running evaluation

To run a model on the QA dataset, execute the `inference.py` script with the following arguments:

```
python evaluation.py <task_number> <model_name> <metric_name> <use_table_code>
```

  -	`task_number`: The task number (1, 2, or 3).
  -	`model_name`: The exact model name (refer to the `config` file for available options).
  -	`metric_name` The evaluation metric (refer to the `config` file for available options).
  - `use_table_code` Whether the responses included table code as input (true or false). This is only applicable for Task 1.

The evaluation results will be printed in the console. For metrics like **LLM-Accuracy** or **VQA-MQM**, an additional `.txt` file will be generated, storing the model responses.
