from models.LLaVa_model import LLaVaModel
#from models.deepseek_model import DeepSeekModel
from models.qwen_model import QwenModel

"""
IMPORTANT: To run deepseek, the github repository must be available in the folder.
If it is, uncomment the import line upwards and the return line downwards.
"""

class TextImageLoader:
    @staticmethod
    def load_model(model_name):
        if model_name.lower() == "llava":
            return LLaVaModel()
        if model_name.lower() == "deepseek":
            #return DeepSeekModel()
            return None
        if model_name.lower() == "qwen":
            return QwenModel()
        raise ValueError("Undefined model name received. Please check the model list to identify the correct model name.")