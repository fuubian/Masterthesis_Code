from models.LLaVa_model import LLaVaModel
from models.deepseek_model import DeepSeekModel

class TextImageLoader:
    @staticmethod
    def load_model(model_name):
        if model_name.lower() == "llava":
            return LLaVaModel()
        if model_name.lower() == "deepseek":
            return
        raise ValueError("Undefined model name received. Please check the model list to identify the correct model name.")