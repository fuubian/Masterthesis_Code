# Masterthesis_VQA
This repository contains all code files of my master thesis "A Benchmark for Scientific Figure and Table Understanding with Multimodal LLMs"

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
