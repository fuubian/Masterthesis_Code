import sys
import os
import csv
import config
from utils.text_image_loader import TextImageLoader
from utils.data_loader import DataLoader

QUESTION_PROMPT_TEMPLATE = """
Answer the following question regarding a scientific {object} in a short way:
{object} caption: {caption}
{table_code_filler}
Question: {question}
"""

def run_inference(model_name, model, task_number, include_table_code, data):
    """
    This functions prompts the QA-pairs to a model and stores it responses in a separate csv file.

    Args:
        model_name (str): The name of the model.
        model (TextImageModel): The initialized object instance of the model.
        task_number (int): The number of the executed task. Value must be in range [1,3].
        include_table_code (bool): Whether the code of tables is also prompted as an input.
        data (dict): A dictionary containing all relevant data to run inference on the model.
    """
    # Open output file
    output_path = os.path.join(config.TASK1_OUTPUT, model_name + "_responses.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        csv_writer = csv.writer(output_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        # Run inference for each QA-pair
        for object_id in data:
            prompt, question, image_path = get_correct_prompt(task_number, object_id, data[object_id, include_table_code])

            try:
                model_response = model.generate_answer(prompt, image_path)
            except Exception as e:
                model_response = f"The model was not able to produce an answer: {e}"
            else:
                csv_writer.writerow([object_id, question, model_response])

    # Print message
    print(f"Inference for task {task_number} has been completed.")

def get_correct_prompt(task_number, object_id, row_data, include_table_code):
    """
    This method identifies which functions need to be executed in order to obtain the correct prompt for the corresponding task.

    Args:
        task_number (int): The number of the executed task. Value must be in range [1,3].
        object_id (str): The id of the table or figure.
        row_data (list): A list containing all relevant information from a QA-pair to run inference on it.
        include_table_code (bool): Whether the code of tables is also prompted as an input.

    Returns:
        (str, str, str): A tuple containing the modified prompt, the input question, and the path to the image file.
    """
    if task_number == 1:
        return modify_prompt_task1(object_id, row_data, include_table_code)
    raise ValueError(f"Unexpected task_number received: {task_number}. A value in range 1-3 was expected.")

def modify_prompt_task1(object_id, row_data, include_table_code):
    """
    This method produces specific prompt for a given QA-pair in task 1.

    Args:
        object_id (str): The id of the table or figure.
        row_data (list): A list containing all relevant information from a QA-pair to run inference on it.
        include_table_code (bool): Whether the code of tables is also prompted as an input.

    Returns:
        (str, str, str): A tuple containing the modified prompt, the input question, and the path to the image file.
    """
    object_type = "Table" if config.TABLE_NAME_FORMAT in object_id else "Figure"
    question = row_data[0]
    image_path = row_data[2]
    caption = row_data[3]

    prompt = QUESTION_PROMPT_TEMPLATE.replace("{object}", object_type).replace("{caption}", caption).replace("{question}", question)
    if include_table_code and (object_type == "Table" or object_type == "Table_02"):
        prompt = prompt.replace("{table_code_filler}", f"Table code: {row_data[4]}")
    else:
        prompt = prompt.replace("{table_code_filler}", "")

    return prompt, question, image_path

def load_task_data(task_number, include_table_code):
    """
    This method identifies the needed data based on the task number and returns it.

    Args:
        task_number (int): The number of the executed task. Value must be in range [1,3].
        include_table_code (bool): Whether the code of tables is also prompted as an input.

    Returns:
        dict: A dictionary containing all relevant data to run inference on the model.
    """
    if task_number == 1:
        return DataLoader.load_qa_test_data(include_table_code)
    raise ValueError(f"Unexpected task_number received: {task_number}. A value in range 1-3 was expected.")

def main(model_name, task_number, include_table_code):
    """
    This main function loads the model and data. Afterwards, the inference on all data is executed.

    Args:
        model_name (str): The name of the model.
        task_number (int): The number of the executed task. Value must be in range [1,3].
        include_table_code (bool): Whether the code of tables is also prompted as an input.
    """
    model = TextImageLoader.load_model(model_name)
    data = load_task_data(task_number, include_table_code)
    print("Model and data were successfully loaded.")

    run_inference(model_name, model, include_table_code, data)

if __name__ == '__main__':
    NUMBER_OF_ARGUMENTS = 3
    args = sys.argv[1:]
    if len(args) != NUMBER_OF_ARGUMENTS:
        print(f"Unexpeceted number of arguments received. Expected: {NUMBER_OF_ARGUMENTS}; Received: {len(args)}")
    else:
        try:
            model_name = args[0]
            task_number = args[1]
            include_table_code = args[2].lower() == "true"
        except ValueError as e:
            print(f"Error occurred while processing arguments: {e}")
        else:
            main(model_name, include_table_code)