import config
from metrics.metric_template import Metric
from utils.data_loader import DataLoader
from bert_score import score

class BertScoreMetric(Metric):
    @staticmethod
    def evaluate(data_dict, model_name):
        categories = {
            "Overall": {"matches": 0, "total": 0},
            "Figure": {"matches": 0, "total": 0},
            "Table": {"matches": 0, "total": 0}
        }

        # Obtain latex-free answers as second reference
        latex_free_dict = DataLoader.read_csv_file(config.QA_TEST_SPLIT_LATEX_CLEAN_PATH, [1])

        # Iterating through all pairs
        for category in categories:
            references = []
            hypothesises = []
            for object_id in data_dict:
                if category == "Overall" or (category == "Figure" and config.FIGURE_NAME_FORMAT in object_id) or (category == "Table" and config.TABLE_NAME_FORMAT in object_id):
                    response= data_dict[object_id][2]
                    lf_reference = latex_free_dict[object_id][0]

                    references.append(lf_reference)
                    hypothesises.append(response)

            if len(references) > 0:
                P, R, F1 = score(hypothesises, references, model_type="microsoft/deberta-xlarge-mnli", lang="en", verbose=True)

                categories[category]["total"] = len(references)
                categories[category]["matches"] = F1.mean().item()

        # Printing results
        print(f"Results of {model_name} with BERTScore:\n")
        for category in categories:
            print(f"{category}: {categories[category]['matches']:.2%} for {categories[category]['total']} objects.")