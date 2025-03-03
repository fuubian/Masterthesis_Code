import sys
import os
from utils.data_loader import DataLoader
from metrics.accuracy import Accuracy
from utils.metric_loader import MetricLoader

def main(task_number, model_name, metric_name):
    # Get output data
    output_path = DataLoader.get_output_path(task_number, model_name)
    if not os.path.exists(output_path):
        print(f"Output file does not exist. Please check the path: {output_path}")
        sys.exit()
    if task_number == 1:
        output_data = DataLoader.read_csv_file(output_path, [1,3])
    elif task_number == 2 or task_number == 3:
        output_data = DataLoader.read_csv_file(output_path, [1,2])
    
    # Apply accuracy for task 2 and 3
    if task_number == 2 or task_number == 3:
        Accuracy.evaluate(output_data)
    
    if task_number == 1:
        metric = MetricLoader.load_metric(metric_name)
        metric.evaluate(output_data)

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 2 or len(args) != 3:
        print(f"Unexpeceted number of arguments received. Expected: 2 or 3; Received: {len(args)}")
    else:
        try:
            task_number = int(args[0])
            model_name = args[1]
            metric_name = None

            if (task_number == 1):
                if len(args) != 3:
                    print(f"Unexpeceted number of arguments received. Expected: 3; Received: {len(args)}")
                    sys.exit()
                metric_name = args[2]
        except ValueError as e:
            print(f"Error occurred while processing arguments: {e}")
        else:
            main(task_number, model_name, metric_name)