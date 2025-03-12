import config
from metrics.metric_template import Metric
from utils.data_loader import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

class BleuMetric(Metric):
    @staticmethod
    def evaluate(data_dict, model_name):
        smoother = SmoothingFunction().method1

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

            response = word_tokenize(data_dict[object_id][2])
            reference = word_tokenize(data_dict[object_id][1])
            lf_reference = word_tokenize(latex_free_dict[object_id][0])

            score = sentence_bleu([reference, lf_reference], response, weights=(1, 0, 0, 0), smoothing_function=smoother)

            categories[category]["total"] += 1
            categories[category]["matches"] += score
        categories["Overall"]["matches"] = categories["Figure"]["matches"] + categories["Table"]["matches"]

        # Printing results
        BleuMetric.print_results(categories, model_name, "BLEU")