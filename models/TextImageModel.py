from abc import ABC, abstractmethod

class TextImageModel(ABC):
    def __init__(self, model_name):
        """
        Initializing the object.

        Args:
            model_name (str): The name of the model so that it can be loaded from HuggingFace.
        """
        self.model_name = model_name

    @abstractmethod
    def generate_answer(self, question_prompt, image):
        """
        Generates an answer for a question based on an image.

        Args:
            question_prompt (str): The prompt containing the question.
            image (Image): The image object on which the question revolves.

        Returns:
            str: The model's response to the prompt.
        """
        return "Default response."