class Metric():
    @staticmethod
    def evaluate(data_dict):
        """
        This function calculates the metric for all instances of a given disc a given task.

        Args:
            data_dict (dict): A dictionary containing for each object_id the correct solution and the model response. For task 1, the question is also included.
        """
        pass

    @staticmethod
    def print_results(categories):
        """
        This function prints the results of a given metric.
        
        Args:
            categories (dict): A dictionary of dictionaries, containing for each category (Overall, Figure, Table) the results.
        """
        print("Results:")
        print("=" * 30)
        for category in categories:
            match_count = categories[category]["matches"]
            total_count = categories[category]["total"]
            print(f"{category:<7}: {match_count:5.2f} / {total_count:<5} -> {match_count / total_count:.2%}")