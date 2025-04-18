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
        important_paths = [config.DATASET_PATH, config.FIGURE_FILES_PATH, config.FIGURE_METADATA_PATH, config.QA_TEST_SPLIT_PATH, config.TABLE_CODE_PATH, 
                           config.QA_TEST_SPLIT_LATEX_CLEAN_PATH, config.TABLE_IMAGE_PATH, config.TABLE_METADATA_PATH, config.QA_TEST_TASK2_PATH, config.QA_TEST_TASK3_PATH]
        for path in important_paths:
            if not os.path.exists(path):
                raise ValueError(f"{path} could not be found. Please check the config file.")
            
    def load_task_data(task_number, include_table_code):
        """
        This method identifies the needed data based on the task number and returns it.

        Args:
            task_number (int): The number of the executed task. Value must be in range [1,3].
            include_table_code (bool): Whether the code of tables is also prompted as an input.

        Returns:
            dict: A dictionary containing all relevant data to run inference on the model.
        """
        if task_number == 1:
            return DataLoader.load_data_task1(include_table_code)
        elif task_number == 2 or task_number == 3:
            return DataLoader.load_data_task23(task_number)
        raise ValueError(f"Unexpected task_number received: {task_number}. A value in range 1-3 was expected.")

    @staticmethod
    def load_data_task1(include_table_code):
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
    def load_data_task23(task_number):
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
    def remove_non_tables(data_dict):
        """
        For the purpose of evaluating VQAs about tables, for which latex code have been used, this function removes all figures from a data dictionary.
        
        Args:
            data_dict (dict): A dictionary containing all model responses and ground truths for both tables and figures.
            
        Returns:
            dict: A dictionary containing all responses and ground truths regarding the VQA task about tables with code.
        """
        table_dict = {}

        for object_id in data_dict:
            if config.TABLE_NAME_FORMAT in object_id:
                table_dict[object_id] = data_dict[object_id]

        return table_dict


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
    
    @staticmethod
    def get_output_path(task_number, model_name):
        """
        This function returns the output path for the corresponding task.

        Args:
            task_number (int): The number of the task. The value must be in range [1,3].
            model_name (str): The name of the model that is used.

        Returns:
            str: The output path to a csv file in which the model responses will be stored.
        """
        output_dir = None
        if task_number == 1:
            output_dir = config.TASK1_OUTPUT
        elif task_number == 2:
            output_dir = config.TASK2_OUTPUT
        elif task_number == 3:
            output_dir = config.TASK3_OUTPUT

        os.path.join(output_dir, model_name + f"_responses_task{task_number}.csv")
        return os.path.join(output_dir, model_name + f"_responses_task{task_number}.csv")
    
    @staticmethod
    def remove_already_inferenced_objects(output_path, data):
        """
        This function removes elements from the QA dictionary which were already processed and are part of the output file.
        
        Args:
            output_path (str): The path to the output file in which the model responses are stored.
            data (dict): A dictionary containing all relevant data to run inference on the model.

        Returns:
            dict: The updated version of the dict without already-processed elements.
        """
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as former_output:
                csv_reader = csv.reader(former_output, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

                for row in csv_reader:
                    if row[0] in data:
                        del data[row[0]]
        return data
    
    @staticmethod 
    def remove_figure_inferenced_objects(output_path, data):
        """
        This function removes figures from the QA dictionary which were already processed in the setting without table_code.
        Additionally, it creates a new csv file containg already infereced figure rows.
        
        Args:
            output_path (str): The path to the output file in which the former model responses are stored.
            data (dict): A dictionary containing all relevant data to run inference on the model.

        Returns:
            str: The path to the new output_file for the table_code setting.
            dict: The updated version of the dict without already-processed figures.
        """
        new_output_path = output_path.replace(".csv", "_code.csv")
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as former_output:
                csv_reader = csv.reader(former_output, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

                with open(new_output_path, "w", newline="", encoding="utf-8") as new_output:
                    csv_writer = csv.writer(new_output, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

                    for row in csv_reader:
                        if config.FIGURE_NAME_FORMAT in row[0] and row[0] in data:
                            csv_writer.writerow(row)
                            del data[row[0]]

        return new_output_path, data