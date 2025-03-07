class Metric():
    @staticmethod
    def evaluate(data_dict, model_name):
        """
        This function calculates the metric for all instances of a given disc a given task.

        Args:
            data_dict (dict): A dictionary containing for each object_id the correct solution and the model response. For task 1, the question is also included.
            model_name (str): The name of the model that is evaluated.
        """
        pass

    @staticmethod
    def print_results(categories, model_name):
        """
        This function prints the results of a given metric.
        
        Args:
            categories (dict): A dictionary of dictionaries, containing for each category (Overall, Figure, Table) the results.
        """
        print(f"Results for {model_name}:")
        print("=" * 30)
        for category in categories:
            match_count = categories[category]["matches"]
            total_count = categories[category]["total"]
            partion = match_count / total_count if total_count > 0 else 0
            print(f"{category:<7}: {match_count:<5} / {total_count:<5} -> {partion:.2%}")