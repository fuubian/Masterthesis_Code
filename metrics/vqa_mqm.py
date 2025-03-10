import config
import regex as re
from metrics.metric_template import Metric
from transformers import AutoModelForCausalLM, AutoTokenizer

class VQA_MQM(Metric):

    # Error types
    error_types = {
        "Critical Value Error": "All values given in the response are significantly different.",
        "Critical Reasoning Error": "The response presents reasoning that is fundamentally different from the reference.",
        "Major Completeness Error": "The response misses relevant information from the reference.",
        "Major Value Error": "One of multiple values is significantly different, affecting correctness.",
        "Major Ambiguity Error": "The response is unclear or could be interpreted in multiple ways.",
        "Major Reasoning Error": "The response presents reasoning that is significantly different from the reference.",
        "Minor Value Error": "A value in the response deviates slightly from the reference but remains within an acceptable range (less than 10% deviation).",
        "Minor Reasoning Error": "The response presents reasoning that is slightly different from the reference.",
        "Minor Completeness Error": "The response misses little information from the reference."
    }

    # Evaluation prompt
    prompt = """
    Evaluate the given model response based on its accuracy and alignment with the question and reference text. Follow these criteria:

    Error Types & Definitions:
        {error_types}
    Acceptable Variations (no penalty):
        - Using synonyms or an alternative phrasing while still conveying the same meaning.
        - Using a different level of detail/specification while still answering the question sufficiently.
        - Using a different unit while providing the same converted result.
        - Minor rounding differences for values.
        - Different formatting.
        - Response using no LaTeX-code while conveying still the same meaning.
        - "Ours" is used as a synonym for "proposed model."
        - Different notation for the same mathematical concept.
        
    IMPORTANT: Don't be strict when identifying errors. Allow minor variations that do not alter the overall meaning. Only classify an error as critical if you are absolutely certain it significantly impacts understanding. If a text span could fall under multiple error types, select only the most relevant one.

    Your output should be of the following format:
    List of errors:
    - [Error Type]: [Brief explanation]

    Your task:
    Evaluate the response for this question:
    Question: {question}
    Reference Text: {reference}
    Response: {response}
    """

    # Modify evaluation prompt to contain error list:
    error_list_string = ""
    for error_type in error_types:
        error_list_string += f"- {error_type}: {error_types[error_type]}\n"
    prompt = prompt.replace("{error_types}", error_list_string)

    @staticmethod
    def evaluate(data_dict, model_name):
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
        filename = model_name + "MQM_evaluation.txt"
        with open(filename, "w", encoding="utf-8") as file_writer:
            for object_id in data_dict:
                is_figure = config.FIGURE_NAME_FORMAT in object_id
                category = "Figure" if is_figure else "Table"
                categories[category]["total"] += 1

                question = data_dict[object_id][0]
                reference = data_dict[object_id][1]
                response = data_dict[object_id][2]
                modified_prompt = VQA_MQM.prompt.replace("{question}", question).replace("{reference}", reference).replace("{response}", response)
                
                try:
                    model_output = VQA_MQM.generateResponse(model, tokenizer, modified_prompt)
                    file_writer.write(object_id + ": " + model_output + "\n")
                    model_output = VQA_MQM.processOutput(model_output)
                except Exception as e:
                    model_output = 0
                    print(f"Model was not able to produce a response: {e}")

                categories[category]["matches"] += model_output
            categories["Overall"]["matches"] = categories["Figure"]["matches"] + categories["Table"]["matches"]

        # Print results
        VQA_MQM.print_results(categories, model_name, "VQA_MQM")

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
            max_new_tokens=400
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    @staticmethod
    def processOutput(output):
        current_score = 1.0
        for type in VQA_MQM.error_types:
            matches = len(re.findall(type, output))

            if matches > 0:
                if "Critical" in type:
                    return 0
                
                penalty_value = 0.5 if "Major" in type else 0.25
                current_score -= penalty_value * matches

                if current_score <= 0:
                    return 0
        
        return current_score