import sys
import os
import csv
import config
import prompt_templates as prompts
from utils.text_image_loader import TextImageLoader
from utils.data_loader import DataLoader

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
    inference_counter = 0
    output_path = os.path.join(config.TASK1_OUTPUT, model_name + f"_responses_task{task_number}.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        csv_writer = csv.writer(output_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        # Run inference for each QA-pair
        for object_id in data:
            prompt, question, solution, image_path = get_correct_prompt(task_number, object_id, data[object_id, include_table_code])

            try:
                model_response = model.generate_answer(prompt, image_path)
            except Exception as e:
                model_response = f"The model was not able to produce an answer: {e}"
            else:
                if question:
                    # Task 1
                    csv_writer.writerow([object_id, question, solution, model_response])
                else:
                    # Task 2 or 3
                    csv_writer.writerow([object_id, solution, model_response])

            # Flushing every 100 elements to reduce risk of data loss
            inference_counter += 1
            if inference_counter % 100 == 0:
                output_file.flush()
                os.fsync(output_file.fileno())

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
        (str, str, str): A tuple containing the modified prompt, the input question, the solution, and the path to the image file.
    """
    if task_number == 1:
        return modify_prompt_task1(object_id, row_data, include_table_code)
    elif task_number == 2 or task_number == 3:
        return modifity_prompt_task23(task_number, object_id, row_data)
    raise ValueError(f"Unexpected task_number received: {task_number}. A value in range 1-3 was expected.")

def modify_prompt_task1(object_id, row_data, include_table_code):
    """
    This method produces the specific prompt for a given QA-pair in task 1.

    Args:
        object_id (str): The id of the table or figure.
        row_data (list): A list containing all relevant information from a QA-pair to run inference on it.
        include_table_code (bool): Whether the code of tables is also prompted as an input.

    Returns:
        (str, str, str, str): A tuple containing the modified prompt, the input question, the solution, and the path to the image file.
    """
    # Retrieve data
    object_type = "Table" if config.TABLE_NAME_FORMAT in object_id else "Figure"
    question = row_data[0]
    solution = row_data[1]
    image_path = row_data[2]
    caption = row_data[3]

    # Modify the prompt
    prompt = prompts.PROMPT_TEMPLATE_TASK1.replace("{object}", object_type).replace("{caption}", caption).replace("{question}", question)
    if include_table_code and (object_type == "Table" or object_type == "Table_02"):
        prompt = prompt.replace("{table_code_filler}", f"Table code: {row_data[4]}")
    else:
        prompt = prompt.replace("{table_code_filler}", "")

    return prompt, question, solution, image_path

def modifity_prompt_task23(task_number, object_id, row_data):
    """
    This method produces the specific prompt for a given QA-pair in task 2 or 3.

    Args:
        object_id (str): The id of the table or figure.
        row_data (list): A list containing all relevant information from a QA-pair to run inference on it.
        include_table_code (bool): Whether the code of tables is also prompted as an input.

    Returns:
        (str, None, str, str): A tuple containing the modified prompt, a placeholder instead of the question, the solution, and the path to the image file.
    """
    # Retrieve data
    object_type = "Table" if config.TABLE_NAME_FORMAT in object_id else "Figure"
    answer_options = row_data[:4]
    solution = convert_number_to_letter(row_data[4])
    image_path = row_data[5]

    # Retrieve the correct question based on task number
    question = None
    if task_number == 2:
        question = prompts.QUESTION_TASK2.replace("{object}", object_type)
    elif task_number == 3:
        question = prompts.QUESTION_TASK3.replace("{object}", object_type)

    # Modify the prompt
    prompt = prompts.PROMPT_TEMPLATE_TASK23.replace("{question}", question)
    for x in range(3):
        prompt = prompt.replace(f"{{option{x+1}}}", answer_options[x])

    return prompt, None, solution, image_path

def convert_number_to_letter(number):
    """
    For multiple choice tasks, this function converts a number into its letter equivalent.
    
    Args:
        number (int): The number indicating an answer option. This value must be in range [1,4].
        
    Returns:
        char: The corresponding letter to the answer option.
    """
    if number == 1:
        return 'A'
    if number == 2:
        return 'B'
    if number == 3:
        return 'C' 
    if number == 4:
        return 'D'
    raise ValueError(f"Unexpected value received: {number}. Number must be in range [1,4].") 

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
    elif task_number == 2 or task_number == 3:
        return DataLoader.load_task_data(task_number)
    raise ValueError(f"Unexpected task_number received: {task_number}. A value in range 1-3 was expected.")

def create_output_dirs():
    """
    This function creates the directories in which the ouput file shall be stored.
    """
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)
    os.makedirs(config.TASK1_OUTPUT, exist_ok=True)
    os.makedirs(config.TASK2_OUTPUT, exist_ok=True)
    os.makedirs(config.TASK3_OUTPUT, exist_ok=True)

def main(task_number, model_name, include_table_code):
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

    create_output_dirs()
    run_inference(model_name, model, task_number, include_table_code, data)

if __name__ == '__main__':
    NUMBER_OF_ARGUMENTS = 3
    args = sys.argv[1:]
    if len(args) != NUMBER_OF_ARGUMENTS:
        print(f"Unexpeceted number of arguments received. Expected: {NUMBER_OF_ARGUMENTS}; Received: {len(args)}")
    else:
        try:
            task_number = int(args[0])
            model_name = args[1]
            include_table_code = args[2].lower() == "true"
        except ValueError as e:
            print(f"Error occurred while processing arguments: {e}")
        else:
            main(task_number, model_name, include_table_code)