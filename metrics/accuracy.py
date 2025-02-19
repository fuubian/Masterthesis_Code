import config

class Accuracy:
    @staticmethod
    def exact_match(data_dict):
        """
        This function calculates and prints the exact match accuracy for task 2 and 3.

        Args:
            data_dict (dict): A dictionary containing for each object_id the correct solution and the model's response.
        """
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
            if data_dict[object_id][0] == data_dict[object_id][1]:
                categories[category]["matches"] += 1
        categories["Overall"]["matches"] = categories["Figure"]["matches"] + categories["Table"]["matches"]

        # Printing results
        print("Results:")
        print("=" * 30)
        for category in categories:
            match_count = categories[category]["matches"]
            total_count = categories[category]["total"]
            print(f"{category:<7}: {match_count:>5} / {total_count:<5} -> {match_count / total_count:.2%}")