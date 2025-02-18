import os
import config
import csv
import random
import regex as re

class DataLoader:
    
    @staticmethod
    def check_data_availability():
        """
        This method checks if the dataset files, which are set in the config file, exist. Otherwise, a ValueError is raised.
        """        
        important_paths = [config.DATASET_PATH, config.FIGURE_FILES_PATH, config.FIGURE_METADATA_PATH, config.QA_TEST_SPLIT_PATH, config.QA_TRAIN_SPLIT_PATH,
                        config.TABLE_CODE_PATH, config.TABLE_IMAGE_PATH, config.TABLE_METADATA_PATH, config.QA_TEST_TASK2_PATH, config.QA_TEST_TASK3_PATH]
        for path in important_paths:
            if not os.path.exists(path):
                raise ValueError(f"{path} could not be found. Please check the config file.")
            
    @staticmethod
    def load_qa_test_data(include_table_code):
        """
        This methods loads the test split data of task 1.
        
        Args:
            include_table_code (bool): Whether the code of tables is also prompted as an input.

        Returns:
            dict: A dictionary containing all relevant data to run inference on the model for task 1.
        """
        DataLoader.check_data_availability()

        # Load QA-pairs in dictionary
        qa_test_split = DataLoader.read_csv_file(config.QA_TEST_SPLIT_PATH, columns=[2,3])
        object_ids = set(qa_test_split.keys())

        # Load image paths
        for object_id in object_ids:
            qa_test_split[object_id].append(DataLoader.get_image_path(object_id))

        # Load captions
        figure_captions = DataLoader.read_csv_file(config.FIGURE_METADATA_PATH, columns=[3], selected_ids=object_ids)
        table_captions = DataLoader.read_csv_file(config.TABLE_METADATA_PATH, columns=[3], selected_ids=object_ids)
        for object_id in object_ids:
            if config.TABLE_NAME_FORMAT in object_id:
                qa_test_split[object_id].append(table_captions[object_id][0])
            else:
                qa_test_split[object_id].append(figure_captions[object_id][0])

        # Load table code
        if include_table_code:
            for object_id in object_ids:
                if config.TABLE_NAME_FORMAT in object_id:
                    qa_test_split[object_id].append(DataLoader.get_table_code(object_id))
                else:
                    qa_test_split[object_id].append(None)

        return qa_test_split
    
    @staticmethod
    def load_task_data(task_number):
        """
        This functions load the test data for either task 2 or task 3.

        Args:
            task_number (int): The number of the task. This value must be either 2 or 3.

        Returns:
            dict: A dictionary containing the answer options, the correct solution, and the image path.
        """
        DataLoader.check_data_availability()

        file_path = None
        if task_number == 2:
            file_path = config.QA_TEST_TASK2_PATH
        elif task_number == 3:
            file_path = config.QA_TEST_TASK3_PATH
        else:
            raise ValueError(f"Unexpeceted task number received. Value must be 2 or 3.")
        
        data_dict = DataLoader.read_csv_file(file_path, [1,2,3,4])

        # Shuffle answer options
        for object_id in data_dict:
            answer_options = data_dict[object_id]
            indexed_list = list(enumerate(answer_options))

            random.shuffle(indexed_list)
            shuffled_list = [val for _, val in indexed_list]
            correct_answer = next(i for i, val in enumerate(shuffled_list) if val == answer_options[0])+1

            data_dict[object_id] = shuffled_list
            data_dict[object_id].append(correct_answer)

        # Load image paths
        for object_id in data_dict:
            data_dict[object_id].append(DataLoader.get_image_path(object_id))

        return data_dict

    @staticmethod
    def read_csv_file(path, columns, selected_ids=None):
        """
        This function opens a csv file and extracts relevant information into a dictionary.

        Args:
            path (str): The file path to the csv file.
            columns (list[int]): A list of all columns that contain information that shall be extracted.
            selected_ids (list[str]): A list of object ids to specify from which selected rows information shall be stored. Default is None.

        Returns:
            dict: A dictionary containing all extracted information for each object id.
        """
        output_dict = {}

        with open(path, "r", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            for row in csv_reader:
                row_id = row[0]
                if not selected_ids or row_id in selected_ids:
                    content = []
                    for column_id in columns:
                        content.append(row[column_id])
                    output_dict[row_id]= content

        return output_dict
    
    @staticmethod
    def get_image_path(object_id):
        """
        This functions derivates the exact file path to an image based on the corresponding object id.
        
        Args:
            object_id (str): The id of either a table or a figure.
            
        Returns:
            str: The path to the image file.
        """
        object_path = None
        if config.TABLE_NAME_FORMAT in object_id:
            object_path = os.path.join(config.TABLE_IMAGE_PATH, object_id + ".png")
        elif config.FIGURE_NAME_FORMAT in object_id:
            object_path = os.path.join(config.FIGURE_FILES_PATH, object_id + ".png")
        else:
            raise ValueError(f"Unknown object type received: {object_id}")
        
        if os.path.exists(object_path):
            return object_path
        raise ValueError(f"Object path was not found: {object_path}")
    
    @staticmethod
    def get_table_code(object_id):
        """
        This function returns the latex code of a table by locating its corresponding file within the dataset.

        Args:
            object_id (str): The id of the table.

        Returns:
            str: The latex code of the table.
        """
        code_path = os.path.join(config.TABLE_CODE_PATH, object_id + ".tex")
        file = open(code_path, "r", encoding="utf-8")
        table_code = file.read()
        file.close()

        # Regex to identify the remove the preamble
        found_tables = re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", table_code, re.DOTALL)
        if len(found_tables) != 1:
            raise ValueError(f"Unexpected occurrence of pagenumbering. Please check manually {object_id}")
        table_code = found_tables[0]

        return table_code