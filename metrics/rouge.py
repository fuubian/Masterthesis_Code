import config
from metrics.metric_template import Metric
from utils.data_loader import DataLoader
from rouge_score import rouge_scorer

class RougeMetric(Metric):
    @staticmethod
    def evaluate(data_dict, model_name):
        USED_VARIANT = "rougeL" # rouge1, rouge2, or rougeL

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

            response = data_dict[object_id][2]
            reference = latex_free_dict[object_id][0]

            scorer = rouge_scorer.RougeScorer([USED_VARIANT], use_stemmer=True)
            scores = scorer.score(reference, response)

            categories[category]["total"] += 1
            categories[category]["matches"] += scores[USED_VARIANT].fmeasure
        categories["Overall"]["matches"] = categories["Figure"]["matches"] + categories["Table"]["matches"]

        # Printing results
        RougeMetric.print_results(categories, model_name, "Rouge")