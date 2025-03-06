import config
from metrics.metric_template import Metric
from utils.data_loader import DataLoader
from pycocoevalcap.cider.cider import Cider

class CiderMetric(Metric):
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
        for category in categories:
            references = {}
            hypothesises = {}
            current_index = 0
            for object_id in data_dict:
                if category == "Overall" or (category == "Figure" and config.FIGURE_NAME_FORMAT in object_id) or (category == "Table" and config.TABLE_NAME_FORMAT in object_id):
                    reference_tokens = data_dict[object_id][1]
                    response_tokens = data_dict[object_id][2]
                    lf_reference_tokens = latex_free_dict[object_id][0]

                    hypothesises[current_index] =  [response_tokens]
                    references[current_index] = [lf_reference_tokens, reference_tokens]
                    current_index += 1

            cider_scorer = Cider()
            score, _ = cider_scorer.compute_score(references, hypothesises)

            categories[category]["total"] = current_index
            categories[category]["matches"] = score

        # Printing results
        print("Results of the evaluation with cider:\n")
        for category in categories:
            print(f"{category}: {categories[category]['matches']:.2%} for {categories[category]['total']} objects.")