import config
import regex as re
from metrics.metric_template import Metric

from models.text_image_model import TextImageModel
from utils.token_loader import TokenLoader
from openai import OpenAI

class LLMAccuracy(Metric):

    # Evaluation prompt
    prompt = """
    Evaluate the response for the following VQA task with respect to the reference text.
    The evaluation criterion is whether the reference and response convey the same meaning.
    
    Your score should consist of one of these three values:
    0   - The response is not correct.
    0.5 - The response is partially correct.
    1   - The response is completely correct.
    
    Question: {question}
    Reference: {reference}
    Response: {response}
    
    Score: 
    """

    @staticmethod
    def evaluate(data_dict, model_name):
        categories = {
            "Overall": {"matches": 0, "total": len(data_dict)},
            "Figure": {"matches": 0, "total": 0},
            "Table": {"matches": 0, "total": 0}
        }

        # Load LLM
        client = OpenAI(api_key=TokenLoader.load_api_key_openai())

        # Iterating through all pairs
        filename = model_name + "_LLM_Acc_evaluation.txt"
        with open(filename, "w", encoding="utf-8") as file_writer:
            for object_id in data_dict:
                is_figure = config.FIGURE_NAME_FORMAT in object_id
                category = "Figure" if is_figure else "Table"
                categories[category]["total"] += 1

                question = data_dict[object_id][0]
                reference = data_dict[object_id][1]
                response = data_dict[object_id][2]
                modified_prompt = LLMAccuracy.prompt.replace("{question}", question).replace("{reference}", reference).replace("{response}", response)
                
                try:
                    model_output = LLMAccuracy.generateResponse(client, modified_prompt)
                    file_writer.write(object_id + ": " + model_output + "\n")
                    model_output = LLMAccuracy.processOutput(model_output)
                except Exception as e:
                    model_output = 0
                    print(f"Model was not able to produce a response: {e}")

                categories[category]["matches"] += model_output
            categories["Overall"]["matches"] = categories["Figure"]["matches"] + categories["Table"]["matches"]

        # Print results
        LLMAccuracy.print_results(categories, model_name, "LLM_Accuracy")

    @staticmethod
    def generateResponse(client, modified_prompt):
        model_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": modified_prompt
                }
            ]
        )
        return model_response
    
    @staticmethod
    def processOutput(output):
        regex_matches = re.findall(r"Score: (0\.5|0|1)", output)
        if len(regex_matches) == 1:
            return float(regex_matches[0])
        return 0