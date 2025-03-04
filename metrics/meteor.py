import config
from metrics.metric_template import Metric
from utils.data_loader import DataLoader
from nltk.translate.meteor_score import meteor_score

class Meteor(Metric):
    @staticmethod
    def evaluate(data_dict):
        categories = {
            "Overall": {"matches": 0, "total": len(data_dict)},
            "Figure": {"matches": 0, "total": 0},
            "Table": {"matches": 0, "total": 0}
        }

        # Obtain latex-free answers as second reference
        latex_free_dict = DataLoader.read_csv_file(config.QA_TEST_SPLIT_LATEX_CLEAN_PATH, [1])

        # Iterating through all pairs
        for object_id in data_dict:
            is_figure = config.FIGURE_NAME_FORMAT in object_id
            category = "Figure" if is_figure else "Table"

            reference_tokens = data_dict[object_id][1].split()
            response_tokens = data_dict[object_id][2].split()
            lf_reference_tokens = latex_free_dict[object_id][0].split()

            score = meteor_score([reference_tokens, lf_reference_tokens], response_tokens)

            categories[category]["total"] += 1
            categories[category]["matches"] += score
        categories["Overall"]["matches"] = categories["Figure"]["matches"] + categories["Table"]["matches"]

        # Printing results
        Meteor.print_results(categories)