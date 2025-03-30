# Master's Thesis: VQA Benchmark
This repository contains all code files of my master's thesis **A Benchmark for Scientific Figure and Table Understanding with Multimodal LLMs**, written at the University of Mannheim.

This benchmark tests the ability of Multimodal LLMs to reason about scientific figures and tables by applying a Visual Question-Answering (VQA) task. Therefore, the models are prompted questions, given an image and its caption. The model responses are stored in a separate csv file und later evaluated by diverse metrics. This benchmark includes traditional metrics from the field of machine translation (e.g., BLEU, ROUGE, etc.), but proposes also two new LLM-based metrics: LLM-Accuracy and VQA-MQM.

The tables and figures were extracted from scientific articles, which were published on arXiv. Only papers that follow a Creative Commons license were included. The corresponding Question-Answer (QA) pairs were generated using the [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) model. The exact notebooks and scripts that were used to create the datasets can be found in the folder `dataset`. Please note that the paths in those code files are not necessarily up-to-date in case you want to execute them. The same goes for the files in the `notebooks` folder, which mainly exist for data visualization and analysis.

Three VQA-tasks are offered by this benchmark:

- **Task 1: MainVQA** - Prompting the model to answer open-ended questions about the content of scientific tables and figures.
- **Task 2: TitleVQA** - Prompting the model to answer single-choice questions, assigning a table or figure to its origin paper (on the basis of the paper title).
- **Task 3: RefVQA** - Prompting the model to answer single-choice questions, assigning a table or figure to a text passage, which references it. The text passage is the same that was used for the creation of QA-pairs for the MainVQA task.

## Model list

The following models are integrated into this benchmark. This list includes a shorter model name (which is required to run the inference script) and the exact version. Generally, the exact model names to run the scripts can be seen and changed in the `config.py` file.

- GLM: [glm-4v-9b](https://huggingface.co/THUDM/glm-4v-9b)
- GPT4: [GPT-4o](https://openai.com/index/hello-gpt-4o/)
- InstructBlip: [instructblip-flan-t5-xxl](https://huggingface.co/Salesforce/instructblip-flan-t5-xxl)
- InternVL: [InternVL2_5-8B](https://huggingface.co/OpenGVLab/InternVL2_5-8B)
- LLaVA: [llava-v1.6-vicuna-13b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf)
- MiniCPM: [MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6)
- Ovis: [Ovis2-8B](https://huggingface.co/AIDC-AI/Ovis2-8B)
- Paligemma: [paligemma2-10b-mix-448](https://huggingface.co/google/paligemma2-10b-mix-448)
- Qwen: [Qwen2.5-VL-7B-Instruct](Qwen/Qwen2.5-VL-7B-Instruct)

## Metric list

The following metrics for MainVQA are integrated into this benchmark. This list includes the exact name that is required to run the evaluation script. Generally, the exact metric names to run the scripts can be seen and changed in the `config.py` file.

- BERTScore
- BLEU
- CIDEr
- LLM_Accuracy
- Meteor
- ROUGE (ROUGE-L)
- VQA_MQM

The tasks TitleVQA and RefVQA use standard accuracy. Therefore, they don't require (and accept) the specification of a metric.

## Installation

1.) **Clone this repository.**

2.) **Install the requirements:** `pip install -r requirements.txt`\
**Note:** The requirements are very general. For specific models, check the Hugging Face model page to ensure the correct package versions are installed.

3.) **Download the datasets:**

In case you intend to use the test and train split of MainVQA, you need to download the following datasets (>20 GB):
  - [SciFigData](https://huggingface.co/datasets/fuubian/SciFigData)
  - [SciTabData](https://huggingface.co/datasets/fuubian/SciTabData)
  - [SciQAData](https://huggingface.co/datasets/fuubian/SciQAData)

If you only want to run inference without using the train split, it is sufficient to download the following datasets (<2 GB):
  - [SciQAData](https://huggingface.co/datasets/fuubian/SciQAData)
  - [SciFigTabData](https://huggingface.co/datasets/fuubian/SciFigTabSubset)

4.) **Ensure that the directory structure is correct.**

The figure and table datasets contain `.tar` files that must be extracted. Afterwards, the directory structure must look like this:
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

Some models need a key or token to be accessed. If you want to run those models, please ensure that the `.env` file is properly set.

  - For GPT-4:
    - Copy `env.example` to `.env`.
    - Add your OpenAI API key to the `.env` file.
  - For Paligemma:
    - Request access to the model on the Hugging Face model page.
    - Copy `env.example` to `.env`.
    - Add your Hugging Face access token to the `.env` file.

    

## Running inference

To run a model on the QA dataset, execute the `inference.py` script with the following arguments:

```
python inference.py <task_number> <model_name> <use_table_code>
```

  -	`task_number`: The task number (1, 2, or 3).
  -	`model_name`: The exact model name (refer to the `config` file for available options).
  -	`use_table_code`: Whether to include table code as input for answering questions (`true` or `false`). This is only applicable for Task 1. False by default.

The model responses will be written into an output directory. There must be no prior output file in the directory before execution. If there exists an incomplete output file and `use_table_code=false`, the script will only run inference for QA-pairs that are not already included. 

For instance, if you want to run the Qwen model on MainVQA without including table code, you need to run the following line:

```
python inference.py 1 qwen false
```

It is highly recommended to first run the script with `use_table_code=false`. If set to `true`, the script will check the output directory for already inferenced QA-pairs regarding figures. Those will be extracted and included in its own output file.

In case you want to test if a model is runnable and able to generate an answer on your system, you can execute the `test_inference.py` script:

```
python test_inference.py <model_name>
```

This script will verify that the images of the datasets are in the correct folder and prompt the selected model to describe the content of a randomly chosen image.

## Running evaluation

To run a model on the QA dataset, execute the `inference.py` script with the following arguments:

```
python evaluation.py <task_number> <model_name> <metric_name> <use_table_code>
```

  -	`task_number`: The task number (1, 2, or 3).
  -	`model_name`: The exact model name (refer to the `config` file for available options).
  -	`metric_name`: The evaluation metric (refer to the `config` file for available options).
  - `use_table_code`: Whether the responses included table code as input (true or false). This is only applicable for Task 1.

The evaluation results will be printed in the console. It is necessary that the output file that containins the responses of the corresponding model exists within the output directory. For metrics like **LLM-Accuracy** or **VQA-MQM**, an additional `.txt` file will be generated, storing the model responses. If `use_table_code` is set to `true`, the script will only evaluate the responses of QA-pairs regarding tables with code. Responses regarding figures will not be included.

For instance, if you want to evaluate all responses of the Qwen model on MainVQA with LLM-Accuracy, you need to run the following line:

```
python evaluation.py 1 qwen LLM-Accuracy false
```

## Fine-tuning

The folder `finetune_scripts` includes an example file for fine-tuning Paligemma on the training data. Due to resource constraints, a more extensive training was not feasible. However, the script can be enhanced to improve the training stage if desired. Additionally, further scripts can be added to support other models.
