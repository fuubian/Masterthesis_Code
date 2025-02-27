import sys
import os
import csv
import config
from utils.text_image_loader import TextImageLoader
from utils.data_loader import DataLoader
from utils.prompt_loader import PromptLoader

MAX_NEW_TOKENS_TASK23 = 10

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
    output_path = DataLoader.get_output_path(task_number, model_name)
    data = DataLoader.remove_already_inferenced_objects(output_path, data)
    with open(output_path, "a", newline="", encoding="utf-8") as output_file:
        csv_writer = csv.writer(output_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        # Run inference for each QA-pair
        print(f"Inference started for {len(data)} objects.")
        for object_id in data:
            prompt, question, solution, image_path = PromptLoader.get_correct_prompt(task_number, object_id, data[object_id], include_table_code)

            try:
                model_response = model.generate_answer(prompt, image_path)
                model_response = model_response.replace(";", ",").replace("|", "-").replace("\n", " ")   
            except Exception as e:
                print(f"The model was not able to produce an answer: {e}")
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
        return DataLoader.load_data_task1(include_table_code)
    elif task_number == 2 or task_number == 3:
        return DataLoader.load_data_task23(task_number)
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