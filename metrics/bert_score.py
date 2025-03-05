import config
from metrics.metric_template import Metric
from utils.data_loader import DataLoader
from bert_score import score

class BertScoreMetric(Metric):
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

            reference = data_dict[object_id][1]
            response= data_dict[object_id][2]
            lf_reference = latex_free_dict[object_id][0]

            references = [reference, lf_reference]
            hypothesis = [response]

            P, R, F1 = score(hypothesis, references, lang="en", verbose=True)

            categories[category]["total"] += 1
            categories[category]["matches"] += F1
        categories["Overall"]["matches"] = categories["Figure"]["matches"] + categories["Table"]["matches"]

        # Printing results
        BertScoreMetric.print_results(categories)