import os
import config
import csv
import regex as re

class DataLoader:
    
    @staticmethod
    def check_data_availability():
        """
        This method checks if the dataset files, which are set in the config file, exist. Otherwise, a ValueError is raised.
        """
        if not os.path.exists(config.DATASET_PATH):
            raise ValueError("No dataset directory was found. Please configure the config file and ensure its existence.")
        
        important_paths = [config.FIGURE_FILES_PATH, config.FIGURE_METADATA_PATH, config.QA_TEST_SPLIT_PATH, config.QA_TRAIN_SPLIT_PATH,
                        config.TABLE_CODE_PATH, config.TABLE_IMAGE_PATH, config.TABLE_METADATA_PATH]
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
            object_path = os.path.join(config.TABLE_IMAGE_PATH, object_id)
        elif config.FIGURE_NAME_FORMAT in object_id:
            object_path = os.path.join(config.FIGURE_FILES_PATH)
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
        code_path = os.path.join(config.TABLE_CODE_PATH, object_id)
        file = open(code_path, "r", encoding="utf-8")
        table_code = file.read()
        file.close()

        # Regex to identify the remove the preamble
        found_tables = re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", table_code, re.DOTALL)
        if len(found_tables) != 1:
            raise ValueError(f"Unexpected occurrence of pagenumbering. Please check manually {object_id}")
        table_code = found_tables[0]

        return table_code