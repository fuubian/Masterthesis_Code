import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data")

# Paths to splits of the main QA-pair dataset
QA_TEST_SPLIT_PATH = os.path.join(DATASET_PATH, "test_split.csv")
QA_TRAIN_SPLIT_PATH = os.path.join(DATASET_PATH, "train_split.csv")

# Paths to cvs files for task 2 and 3
QA_TEST_TASK2_PATH = os.path.join(DATASET_PATH, "qa_pairs_task2.csv")
QA_TEST_TASK3_PATH = os.path.join(DATASET_PATH, "qa_pairs_task3.csv")

# Metadata paths to csv file
FIGURE_METADATA_PATH = os.path.join(DATASET_PATH, "figures.csv")
TABLE_METADATA_PATH = os.path.join(DATASET_PATH, "tables.csv")

# Paths to figure and table files
FIGURE_FILES_PATH = os.path.join(DATASET_PATH, "figures")
TABLE_IMAGE_PATH = os.path.join(DATASET_PATH, "tables", "image")
TABLE_CODE_PATH = os.path.join(DATASET_PATH, "tables", "code")

# The name formats for table and figure files
TABLE_NAME_FORMAT = "_TAB_"
FIGURE_NAME_FORMAT = "_FIG_"

# Output paths
OUTPUT_PATH = os.path.join(BASE_DIR, "output")
TASK1_OUTPUT = os.path.join(OUTPUT_PATH, "task1")
TASK2_OUTPUT = os.path.join(OUTPUT_PATH, "task2")
TASK3_OUTPUT = os.path.join(OUTPUT_PATH, "task3")

# Model list
LLAVA_MODEL_NAME = "llava"
DEEPSEEK_MODEL_NAME = "deepseek"
QWEN_MODEL_NAME = "qwen"
INSTRCUCTBLIP_MODEL_NAME = "blip"
OVIS_MODEL_NAME = "ovis"
PALIGEMMA_MODEL_NAME = "paligemma"
PIXTRAL_MODEL_NAME = "pixtral"
INTERNVL_MODEL_NAME = "internvl"