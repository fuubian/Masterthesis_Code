import config
import prompt_templates as prompts
from data_loader import DataLoader

class PromptLoader:
    @staticmethod
    def get_correct_prompt(task_number, object_id, row_data, include_table_code):
        """
        This method identifies which functions need to be executed in order to obtain the correct prompt for the corresponding task.

        Args:
            task_number (int): The number of the executed task. Value must be in range [1,3].
            object_id (str): The id of the table or figure.
            row_data (list): A list containing all relevant information from a QA-pair to run inference on it.
            include_table_code (bool): Whether the code of tables is also prompted as an input.

        Returns:
            (str, str, str): A tuple containing the modified prompt, the input question, the solution, and the path to the image file.
        """
        if task_number == 1:
            return PromptLoader.modify_prompt_task1(object_id, row_data, include_table_code)
        elif task_number == 2 or task_number == 3:
            return PromptLoader.modifity_prompt_task23(task_number, object_id, row_data)
        raise ValueError(f"Unexpected task_number received: {task_number}. A value in range 1-3 was expected.")

    @staticmethod
    def modify_prompt_task1(object_id, row_data, include_table_code):
        """
        This method produces the specific prompt for a given QA-pair in task 1.

        Args:
            object_id (str): The id of the table or figure.
            row_data (list): A list containing all relevant information from a QA-pair to run inference on it.
            include_table_code (bool): Whether the code of tables is also prompted as an input.

        Returns:
            (str, str, str, str): A tuple containing the modified prompt, the input question, the solution, and the path to the image file.
        """
        # Retrieve data
        object_type = "Table" if config.TABLE_NAME_FORMAT in object_id else "Figure"
        question = row_data[0]
        solution = row_data[1]
        image_path = row_data[2]
        caption = row_data[3]

        # Modify the prompt
        prompt = prompts.PROMPT_TEMPLATE_TASK1.replace("{object}", object_type).replace("{caption}", caption).replace("{question}", question)
        if include_table_code and (object_type == "Table" or object_type == "Table_02"):
            prompt = prompt.replace("{table_code_filler}", f"Table code: {row_data[4]}")
        else:
            prompt = prompt.replace("{table_code_filler}", "")

        return prompt, question, solution, image_path
    
    def modifity_prompt_task23(task_number, object_id, row_data):
        """
        This method produces the specific prompt for a given QA-pair in task 2 or 3.

        Args:
            object_id (str): The id of the table or figure.
            row_data (list): A list containing all relevant information from a QA-pair to run inference on it.
            include_table_code (bool): Whether the code of tables is also prompted as an input.

        Returns:
            (str, None, str, str): A tuple containing the modified prompt, a placeholder instead of the question, the solution, and the path to the image file.
        """
        # Retrieve data
        object_type = "Table" if config.TABLE_NAME_FORMAT in object_id else "Figure"
        answer_options = row_data[:4]
        solution = DataLoader.convert_number_to_letter(row_data[4])
        image_path = row_data[5]

        # Retrieve the correct question based on task number
        question = None
        if task_number == 2:
            question = prompts.QUESTION_TASK2.replace("{object}", object_type)
        elif task_number == 3:
            question = prompts.QUESTION_TASK3.replace("{object}", object_type)

        # Modify the prompt
        prompt = prompts.PROMPT_TEMPLATE_TASK23.replace("{question}", question)
        for x in range(4):
            prompt = prompt.replace(f"{{option{x+1}}}", answer_options[x])

        return prompt, None, solution, image_path
    
    @staticmethod
    def convert_number_to_letter(number):
        """
        For multiple choice tasks, this function converts a number into its letter equivalent.
        
        Args:
            number (int): The number indicating an answer option. This value must be in range [1,4].
            
        Returns:
            char: The corresponding letter to the answer option.
        """
        if number == 1:
            return 'A'
        if number == 2:
            return 'B'
        if number == 3:
            return 'C' 
        if number == 4:
            return 'D'
        raise ValueError(f"Unexpected value received: {number}. Number must be in range [1,4].")