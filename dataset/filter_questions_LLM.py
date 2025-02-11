from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import csv
import os
import sys
import regex as re

# Variables and directories
workspace_dir = "/pfs/work7/workspace/scratch/ma_frajwa-dataset/"
qa_pair_file = workspace_dir + "qa_pairs.csv"
result_file = workspace_dir + "filtering_pairs.csv"

# Prompt
validation_prompt = """
For the following Question-Answer pair related to a table or figure, determine whether the question is well-formed 
and meaningful, and whether the provided answer is logically and factually correct. Respond with a simple 'Yes' or 'No'.
Question: {question}
Answer: {answer}
"""

def get_qa_pairs(file_path, max_number):
    """
    Extracts QA-pairs from the corresponding csv file.

    Args:
        file_path (str): Path to the csv file.
        max_number (int): Number of how many qa-pairs should be extracted. If 0, all pairs will be extracted.

    Returns:
        list[(str, str, str)]: A list of tuples, containing the object_id, question and answer.
    """
    qa_pairs = []
    with open(file_path, "r", encoding="utf-8") as input_file:
        spamreader = csv.reader(input_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in spamreader:
            object_id = row[0]
            question = row[2]
            answer = row[3]
            qa_pairs.append((object_id, question, answer))
    
    return qa_pairs

def generate_response(question, answer, model, tokenizer):
    """
    Generate a response of the LLM to decide if a QA-pair is valid.

    Args:
        question (str): The question of the QA-pair.
        answer (str): The answer of the QA-pair.
        model (AutoModelForCausalLM): The model used to generate the response.
        tokenizer (AutoTokenizer): The tokenizer used to encode the prompt.

    Returns:
        response (str): The generated response by the model.
    """
    input_prompt = validation_prompt.replace("{question}", question).replace("{answer}", answer)
    
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant that can understand Question-Answer pairs from scientific data."},
        {"role": "user", "content": input_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def main(start_index, max_number):
    """
    Main function. Loads the model, tokenizer and validates QA-pairs. Results will be written into a csv file.

    Args:
        start_index (int): The index of the first QA-pair that shall be validated.
        max_number (int): Number of how many QA-pairs should be used. If 0, all pairs will be used.
    """

    # Loading model
    model_id = "Qwen/Qwen2.5-14B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    qa_pairs = get_qa_pairs(qa_pair_file, max_number)

    # Opening output file
    with open(result_file, "a", encoding="utf-8") as output_file:
        csv_writer = csv.writer(output_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        # Iteratung through QA-pairs
        counter = 0
        for pair in qa_pairs:
            if counter < start_index:
                counter += 1
                continue
            try:
                model_response = generate_response(pair[1], pair[2], model, tokenizer)
            except Exception as e:
                print(f"Error: {e}")
            else:
                csv_writer.writerow([pair[0], pair[1], pair[2], model_response])

                # Flushing every 100 entries
                if counter % 100 == 0:
                    output_file.flush()
                    os.fsync(output_file.fileno())
            counter += 1
            if counter == start_index + max_number:
                break

# Running main function
if __name__ == '__main__':
    NUMBER_OF_ARGUMENTS = 2
    args = sys.argv[1:]
    if len(args) != NUMBER_OF_ARGUMENTS:
        print(f"Unexpected number of arguments received. Expected: {NUMBER_OF_ARGUMENTS}; Received: {len(args)}")
    else:
        try:
            start_index = int(args[0])
            max_number = int(args[1])
        except ValueError as e:
            print(f"Error occurred while processing arguments: {e}")
        else:
            main(start_index, max_number)