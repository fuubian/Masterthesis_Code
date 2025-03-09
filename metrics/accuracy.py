import config
import regex as re
from metrics.metric_template import Metric

class Accuracy(Metric):
    @staticmethod
    def evaluate(data_dict, model_name):
        categories = {
            "Overall": {"matches": 0, "total": len(data_dict)},
            "Figure": {"matches": 0, "total": 0},
            "Table": {"matches": 0, "total": 0}
        }

        # Iterating through all pairs
        for object_id in data_dict:
            is_figure = config.FIGURE_NAME_FORMAT in object_id
            category = "Figure" if is_figure else "Table"

            categories[category]["total"] += 1
            processed_response = Accuracy.process_model_response(data_dict[object_id][1])
            if data_dict[object_id][0] == processed_response:
                categories[category]["matches"] += 1
        categories["Overall"]["matches"] = categories["Figure"]["matches"] + categories["Table"]["matches"]

        # Printing results
        Accuracy.print_results(categories, model_name)

    @staticmethod
    def process_model_response(response):
        """
        This function extracts the single-choice answer (A,B,C,D) from a model's response.

        Args:
            response (str): The response of a model for a single choice task.

        Returns:
            str: The extracted choice by the model. Either 'A', 'B', 'C', 'D', or '-1' if no exact choice could be extracted.
        """
        if len(response) == 1:
            return response
        regex_matches = re.findall(r"[ABCD]\)", response)
        if len(regex_matches) == 1:
            return regex_matches[0][0]
        return response[0]