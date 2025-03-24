import sys
import csv
import json

def csv_to_json(csv_file, json_file):
    """
    This function creates a json file based on a csv file.
    
    Args:
        csv_file (str): The path to the input csv file.
        json_file (str): The path to the output json file.
    """
    data = []
    
    with open(csv_file, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in csv_reader:
            entry = {
                "image_id": row[0],
                "question": row[2],
                "answer": row[3]
            }
            data.append(entry)
    
    with open(json_file, mode='w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)
    
    print(f"JSON file saved as {json_file}")

if __name__ == '__main__':
    NUMBER_OF_ARGUMENTS = 2
    args = sys.argv[1:]
    if len(args) != NUMBER_OF_ARGUMENTS:
        print(f"Unexpeceted number of arguments received. Expected: {NUMBER_OF_ARGUMENTS}; Received: {len(args)}")
    else:
        try:
            csv_file = args[0]
            json_file = args[1]
        except ValueError as e:
            print(f"Error occurred while processing arguments: {e}")
        else:
            csv_to_json(csv_file, json_file)
