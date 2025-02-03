import csv
import os
import re

# Variables and directories
workspace_dir = "/pfs/work7/workspace/scratch/ma_frajwa-dataset/"
qa_output_dir = workspace_dir + "qa_output/"
output_file = workspace_dir + "qa_pairs.csv"
error_file = workspace_dir + "qa_errors.csv"

# File name pattern for tables
table_file_format = "_TAB_"

# Regex patterns for QA-pair identification
question_regex = r"Question: (.*)"
answer_regex = r"Answer: (.*)"

# Get all qa pair files of either tables or figures
def get_files():
    file_collection = os.listdir(qa_output_dir)
    qa_pair_files = set()
    for file in file_collection:
            qa_pair_files.add(file)
    return qa_pair_files

# Extracts and returns the qa-pair from a txt file
def extract_qa_pair(file):
    txt_file = open(qa_output_dir + file, "r", encoding='utf-8')
    file_content = txt_file.read()
    txt_file.close()

    try:
        question = re.findall(question_regex, file_content)[0]
        answer = re.findall(answer_regex, file_content)[0]
    except Exception as e:
        print(f"QA-Pair could not be extracted: {e}")
        return file_content, None
    else:
        return question, answer

# Main function: writes all qa_pairs into a csv file and faulty generated pairs into a separate one
def main():
    file_collection = get_files()

    with open(output_file, "w", newline='', encoding='utf-8') as csv_file_success:
        with open(error_file, "w", newline='', encoding='utf-8') as csv_file_error:
            csv_writer_success = csv.writer(csv_file_success, delimiter=';', quotechar='|', quoting=csv.QUOTE_ALL)
            csv_writer_error = csv.writer(csv_file_error, delimiter=';', quotechar='|', quoting=csv.QUOTE_ALL)

            for file in file_collection:
                question, answer = extract_qa_pair(file)
                object_id = file.replace(".txt", "")
                object_type = None
                if table_file_format in object_id:
                    object_type = "Table"
                else:
                    object_type = "Figure"
                if question and answer:
                    # Successful extraction case
                    csv_writer_success.writerow([object_id, object_type, question, answer])
                else:
                    # Error case
                    csv_writer_error.writerow([object_id, object_type, question])

# Running main function
if __name__ == '__main__':
    main()