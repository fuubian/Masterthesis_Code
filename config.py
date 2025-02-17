import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data")

# Paths to splits of QA-pair dataset
QA_TEST_SPLIT_PATH = os.path.join(DATASET_PATH, "test_split.csv")
QA_TRAIN_SPLIT_PATH = os.path.join(DATASET_PATH, "train_split.csv")

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