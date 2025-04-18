import sys
import os
from utils.data_loader import DataLoader
from metrics.accuracy import Accuracy
from utils.metric_loader import MetricLoader

def main(task_number, model_name, metric_name, evaluate_table_code):
    # Get output data
    output_path = DataLoader.get_output_path(task_number, model_name)
    if not os.path.exists(output_path):
        print(f"Output file does not exist. Please check the path: {output_path}")
        sys.exit()
    if task_number == 1:
        if evaluate_table_code:
            # Evaluating only responses regarding tables with latex code
            output_path = output_path.replace(".csv", "_code.csv")
            output_data = DataLoader.read_csv_file(output_path, [1,2,3])
            output_data = DataLoader.remove_non_tables(output_data)
        else:
            # Evaluating figure and table responses without latex code
            output_data = DataLoader.read_csv_file(output_path, [1,2,3])
    elif task_number == 2 or task_number == 3:
        output_data = DataLoader.read_csv_file(output_path, [1,2])
    
    # Apply accuracy for task 2 and 3
    if task_number == 2 or task_number == 3:
        Accuracy.evaluate(output_data, model_name)
    
    if task_number == 1:
        metric = MetricLoader.load_metric(metric_name)
        metric.evaluate(output_data, model_name)

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 2 and len(args) != 4:
        print(f"Unexpeceted number of arguments received. Expected: 2 or 4; Received: {len(args)}")
    else:
        try:
            task_number = int(args[0])
            model_name = args[1].lower()
            metric_name = None
            evaluate_table_code = False

            if (task_number == 1):
                if len(args) != 4:
                    print(f"Unexpeceted number of arguments received. Expected: 4; Received: {len(args)}")
                    sys.exit()
                metric_name = args[2].lower()
                evaluate_table_code = args[3].lower() == "true"
        except ValueError as e:
            print(f"Error occurred while processing arguments: {e}")
        else:
            main(task_number, model_name, metric_name, evaluate_table_code)