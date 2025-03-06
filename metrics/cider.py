import config
from metrics.metric_template import Metric
from utils.data_loader import DataLoader
from pycocoevalcap.cider.cider import Cider
from nltk.tokenize import word_tokenize

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
                    reference_tokens = word_tokenize(data_dict[object_id][1])
                    response_tokens = word_tokenize(data_dict[object_id][2])
                    lf_reference_tokens = word_tokenize(latex_free_dict[object_id][0])

                    hypothesises[current_index] =  [response_tokens]
                    references[current_index] = [reference_tokens, lf_reference_tokens]
                    current_index += 1

            # Format in lower case
            refs = {i: [r.lower() for r in references[i]] for i in references}
            hyps = {i: [hypothesises[i][0].lower()] for i in hypothesises}

            cider_scorer = Cider()
            score, _ = cider_scorer.compute_score(refs, hyps)

            categories[category]["total"] = current_index
            categories[category]["matches"] = score

        # Printing results
        print("Results of the evaluation with cider:\n")
        for category in categories:
            print(f"{category}: {categories[category]['matches']} for {categories[category]['total']} objects.")