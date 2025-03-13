import os
import csv
import config
import shutil

# Data paths
subset_id_file = "human_evaluation_id.txt"
output_folder = "subset_images"
output_csv_file = "subset_responses.csv"

# Get ids from subset
subset_set = set()
with open(subset_id_file, "r", encoding="utf-8") as txt_file:
    for line in txt_file:
        subset_set.add(line.replace("\n", ""))

# Copy ids into a new folder
os.makedirs(output_folder, exist_ok=True)
for file_id in subset_set:
    if config.FIGURE_NAME_FORMAT in file_id:
        file_path = os.path.join(config.FIGURE_FILES_PATH, file_id+".png")
    elif config.TABLE_NAME_FORMAT in file_id:
        file_path = os.path.join(config.TABLE_IMAGE_PATH, file_id+".png")
    else:
        raise ValueError(f"Unexpected ID obtained: {file_id}")
    
    try:
        shutil.copy(file_path, os.path.join(output_folder, file_id+".png"))
    except Exception as e:
        print(e)

print("Files have been successfully copied to the new location.")

# Create human evaluation csv file for all models
model_output_folder = "output/task1/"

with open(output_csv_file, "w", encoding="utf-8", newline="") as output_csv_file:
    csv_writer = csv.writer(output_csv_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for output_file in os.listdir(model_output_folder):
        if "code" in output_file:
            continue

        model_name = output_file.replace("_responses_task1.csv", "")
        file_path = model_output_folder + output_file
        with open(file_path, "r", encoding="utf-8") as input_csv_file:
            csv_reader = csv.reader(input_csv_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            for row in csv_reader:
                object_id = row[0]
                question = row[1]
                ground_truth = row[2]
                model_response = row[3]

                if object_id in subset_set:
                    csv_writer.writerow[model_name, object_id, question, ground_truth, model_response]

print("csv file with all model responses of the subset was successfully created.")