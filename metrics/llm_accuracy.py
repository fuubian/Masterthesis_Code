import config
import regex as re
from metrics.metric_template import Metric
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    def evaluate(data_dict):
        categories = {
            "Overall": {"matches": 0, "total": len(data_dict)},
            "Figure": {"matches": 0, "total": 0},
            "Table": {"matches": 0, "total": 0}
        }

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

        # Iterating through all pairs
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
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
                    model_output = LLMAccuracy.generateResponse(model, tokenizer, modified_prompt)
                    file_writer.write(object_id + ": " + model_output + "\n")
                    model_output = LLMAccuracy.processOutput(model_output)
                except Exception as e:
                    model_output = 0
                    print(f"Model was not able to produce a response: {e}")

                    categories[category]["matches"] += model_output
            categories["Overall"]["matches"] = categories["Figure"]["matches"] + categories["Table"]["matches"]

        # Print results
        LLMAccuracy.print_results(categories)

    @staticmethod
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
            max_new_tokens=50
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    @staticmethod
    def processOutput(output):
        regex_matches = re.findall(r"Score: (0\.5|0|1)", output)
        if len(regex_matches) == 1:
            return float(regex_matches[0])
        return 0