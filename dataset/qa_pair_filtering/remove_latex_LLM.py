import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.data_loader import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
input_file = os.path.join(PROJECT_ROOT, "data", "test_split.csv")
output_file = os.path.join(PROJECT_ROOT, "data", "latex_clean_test_split.csv")

prompt = """
Extract the plain text from the following input, removing any LaTeX code.
If no LaTeX code is present, return the input unchanged.
Do not provide any explanations do your decision. Only return the plain text in your response.

Input: {input}
Output: 
"""

def generateResponse(model, tokenizer, modified_prompt):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant to evaluate responses of VQA tasks."},
        {"role": "user", "content": f"{modified_prompt}"}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=500
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def main():
    # Load LLM
    model_id = "Qwen/Qwen2.5-14B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        offload_folder="offload_folder",
        offload_state_dict=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print("Model was successfully initialized.")

    # Iterate through test set
    test_set_dict = DataLoader.read_csv_file(input_file, [3])
    print("Iteration through the dataset is starting.")
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        csv_writer = csv.writer(file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for object_id in test_set_dict:
            input_reference = test_set_dict[object_id][0]
            modified_prompt = prompt.replace("{input}", input_reference)
            try:
                output_reference = generateResponse(model, tokenizer, modified_prompt).replace(";", ",").replace("|", "-").replace("\n", " ")
            except Exception as e:
                output_reference = ""
                print(f"No model response received: {e}")

            csv_writer.writerow([object_id, output_reference])

    # Print message
    print("The process was successfully completed.")

if __name__ == '__main__':
    main()