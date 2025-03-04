import config
from metrics.metric_template import Metric
from nltk.translate.meteor_score import meteor_score

class Meteor(Metric):
    @staticmethod
    def evaluate(data_dict):
        categories = {
            "Overall": {"matches": 0, "total": len(data_dict)},
            "Figure": {"matches": 0, "total": 0},
            "Table": {"matches": 0, "total": 0}
        }

        # Iterating through all pairs
        for object_id in data_dict:
            is_figure = config.FIGURE_NAME_FORMAT in object_id
            category = "Figure" if is_figure else "Table"

            reference = Meteor.remove_latex(data_dict[object_id][1])
            response = Meteor.remove_latex(data_dict[object_id][2])
            reference_tokens = reference.split()
            response_tokens = response.split()

            score = meteor_score([reference_tokens], response_tokens)

            categories[category]["total"] += 1
            categories[category]["matches"] += score
        categories["Overall"]["matches"] = categories["Figure"]["matches"] + categories["Table"]["matches"]

        # Printing results
        Meteor.print_results(categories)