import sys
import os
import regex as re
from utils.data_loader import DataLoader
from metrics.accuracy import Accuracy

def process_model_response(response):
    """
    This function extracts the single-choice answer (A,B,C,D) from a model's response.

    Args:
        response (str): The response of a model for a single choice task.

    Returns:
        str: The extracted choice by the model. Either 'A', 'B', 'C', 'D', or '-1' if no exact choice could be extracted.
    """
    regex_matches = re.findall(r"[ABCD]\)", response)
    if len(regex_matches) == 1:
        return regex_matches[0][0]
    return "-1"

def main(task_number, model_name):
    # Get output data
    output_path = DataLoader.get_output_path(task_number, model_name)
    if not os.path.exists(output_path):
        print(f"Output file does not exist. Please check the path: {output_path}")
        sys.exit()
    output_data = DataLoader.read_csv_file(output_path, [1,2])
    
    # Process responses and apply accuracy
    for key in output_data:
        output_data[key][1] = process_model_response(output_data[key][1])
    Accuracy.exact_match(output_data)

if __name__ == '__main__':
    NUMBER_OF_ARGUMENTS = 2
    args = sys.argv[1:]
    if len(args) != NUMBER_OF_ARGUMENTS:
        print(f"Unexpeceted number of arguments received. Expected: {NUMBER_OF_ARGUMENTS}; Received: {len(args)}")
    else:
        try:
            task_number = int(args[0])
            model_name = args[1]
        except ValueError as e:
            print(f"Error occurred while processing arguments: {e}")
        else:
            main(task_number, model_name)