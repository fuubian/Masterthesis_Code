import sys
import config
from utils.text_image_loader import TextImageLoader
from utils.qa_data_loader import QADataLoader

QUESTION_PROMPT_TEMPLATE = """
Answer the following question regarding a scientific {object} in a short way:
{object} caption: {caption}
{table_code_filler}
Question: {question}
"""

def run_inference(model, include_table_code, qa_data):
    for object_id in qa_data:
        object_type = "Table" if config.TABLE_NAME_FORMAT in object_id else "Figure"
        question = qa_data[object_id][0]
        answer = qa_data[object_id][1]
        image_path = qa_data[object_id][2]
        caption = qa_data[object_id][3]

        prompt = QUESTION_PROMPT_TEMPLATE.replace("{object}", object_type).replace("{caption}", caption).replace("{question}", question)
        if include_table_code:
            prompt = prompt.replace("{table_code_filler}", f"Table code: {qa_data[object_id][4]}")

        model_response = model.generate_answer(prompt, image_path)
        #TODO STORE REPONSE IN OUTPUT_FILE


def main(model_name, include_table_code):
    model = TextImageLoader.load_model(model_name)
    qa_data = QADataLoader.load_qa_test_data(include_table_code)
    print("Model and data was successfully loaded.")

    run_inference(model, include_table_code, qa_data)

if __name__ == '__main__':
    NUMBER_OF_ARGUMENTS = 2
    args = sys.argv[1:]
    if len(args) != NUMBER_OF_ARGUMENTS:
        print(f"Unexpeceted number of arguments received. Expected: {NUMBER_OF_ARGUMENTS}; Received: {len(args)}")
    else:
        try:
            model_name = args[0]
            include_table_code = args[1].lower() == "true"
        except ValueError as e:
            print(f"Error occurred while processing arguments: {e}")
        else:
            main(model_name, include_table_code)