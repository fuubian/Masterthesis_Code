import config
import regex as re
from metrics.metric_template import Metric
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

class VQA_MQM(Metric):

    # Evaluation prompt
    prompt = """
    Evaluate the given model response based on its accuracy and alignment with the question and reference text. Follow these criteria:

    Error Types & Definitions:
        - Minor Value Error: One value in the response is slightly different but still within an acceptable range.
        - Major Value Error: One of multiple values is significantly different, affecting correctness.
        - Critical Value Error: All values given in the response are significantly different.
        - Minor Unit Error: The response uses a different unit than the reference.
        - Minor Reasoning Error: The response presents reasoning that is slightly different from the reference.
        - Critical Reasoning Error: The response presents reasoning that is fundamentally different from the reference.
        - Major Completeness Error: The response misses some relevant information from the reference.
        - Critical Completeness Error: The response misses all relevant information from the reference.
        - Minor Ambiguity Error: The response is unclear or could be interpreted in multiple ways.
        - Major Hallucination Error: The response includes additional information that is not present in the reference.

    Acceptable Variations (no penalty):
        - Using synonyms or an alternative phrasing while still conveying the same meaning.
        - Using a different level of detail/specification while still answering the question sufficiently.
        - Minor rounding differences for values.
        - Different formatting.
        - Response using no LaTeX-code while conveying still the same meaning.
        - "Ours" is used as a synonym for "proposed model."
        - Different notation for the same mathematical concept.

    Your output should look like this:
    Score: [your score]
    List of errors:
    - [Error Type]: [Brief explanation]

    Your task:
    Evaluate the response for this question:
    Question: {question}
    Reference Text: {reference}
    Response: {response}
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
        VQA_MQM.print_results(categories)

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
        score_match = re.search(r"Score:\s*(-?\d+(\.\d+)?)", output)
        if score_match:
            score = int(score_match.group(1))
        else:
            print(f"Score was not found for {output}")
            score = 0
        return score