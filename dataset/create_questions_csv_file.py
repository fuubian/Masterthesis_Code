import csv
import os
import re

# Variables and directories
workspace_dir = "/pfs/work7/workspace/scratch/ma_frajwa-dataset/"
qa_output_dir = workspace_dir + "qa_output/"
qa_output_dir_02 = workspace_dir + "qa_output2/"
output_file = workspace_dir + "qa_pairs.csv"
error_file = workspace_dir + "qa_errors.csv"

# File name pattern for tables
table_file_format = "_TAB_"

# Regex patterns for QA-pair identification
question_regex = r"Question: (.*)"
answer_regex = r"Answer: (.*)"

def extract_qa_pair(file_path):
    """
    Extracts a question-answer pair from a txt file.

    Args:
        file_path (str): Path to the txt file containing the QA-pair.

    Returns:
        tuple[str, str]: A tuple containing the question and answer.
    """
    txt_file = open(file_path, "r", encoding='utf-8')
    file_content = txt_file.read()
    txt_file.close()

    try:
        question = re.findall(question_regex, file_content)[0]
        answer = re.findall(answer_regex, file_content)[0]
    except Exception as e:
        print(f"QA-Pair could not be extracted: {e}")
        return file_content.replace("\n", ""), None
    else:
        return question, answer

def main():
    """
    Writes all qa_pairs into a csv file and faulty generated pairs into a separate one.
    """
    file_collection = os.listdir(qa_output_dir)  # Containing qa_pairs from the main generation process
    file_collection_extra = os.listdir(qa_output_dir_02) # Containing qa_pairs from the second generation process without using table code

    with open(output_file, "w", newline='', encoding='utf-8') as csv_file_success:
        with open(error_file, "w", newline='', encoding='utf-8') as csv_file_error:
            csv_writer_success = csv.writer(csv_file_success, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer_error = csv.writer(csv_file_error, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            write_into_csv(file_collection, qa_output_dir, "Table", csv_writer_success, csv_writer_error)
            write_into_csv(file_collection_extra, qa_output_dir_02, "Table_02", csv_writer_success, csv_writer_error)

def write_into_csv(file_collection, qa_dir, table_type, writer_success, writer_error):
    """
    Write QA-pairs from files within a file_collection into a csv file.

    Args:
        file_collection (list): List of all file names that shall be processed.
        qa_dir (str): Path to the directory containing the files.
        table_type (str): Label that should be used for tables.
        writer_success (csv.writer): CSV.Writer to write successful extracted QA-pairs into a csv file.
        writer_error (csv.writer): CSV.Writer to write faulty QA-pairs into a csv file.
    """
    for file in file_collection:
        question, answer = extract_qa_pair(qa_dir + file)
        object_id = file.replace(".txt", "")
        object_type = None
        if table_file_format in object_id:
            object_type = table_type
        else:
            object_type = "Figure"
        if question and answer:
            writer_success.writerow([object_id, object_type, question, answer]) # Successful extraction case
        else:
            writer_error.writerow([object_id, object_type, question]) # Error case

# Running main function
if __name__ == '__main__':
    main()