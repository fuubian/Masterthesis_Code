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
        for object_id in data_dict:
            is_figure = config.FIGURE_NAME_FORMAT in object_id
            category = "Figure" if is_figure else "Table"

            hypothesis = {"0": [data_dict[object_id][2]]} # Model response
            references = {
                "0": [
                    data_dict[object_id][1],
                    latex_free_dict[object_id][0]
                ]  # Multiple reference answers
            }

            # Format the references to be in lower case
            refs = {i: [r.lower() for r in references[i]] for i in references}
            hyps = {i: [hypothesis[i][0].lower()] for i in hypothesis}  # Single hypothesis per image

            cider_scorer = Cider()
            score, _ = cider_scorer.compute_score(refs, hyps)

            categories[category]["total"] += 1
            categories[category]["matches"] += score
        categories["Overall"]["matches"] = categories["Figure"]["matches"] + categories["Table"]["matches"]

        # Printing results
        CiderMetric.print_results(categories)