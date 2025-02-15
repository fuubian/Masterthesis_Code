from models.LLaVa_model import LLaVaModel

class TextImageLoader:
    @staticmethod
    def load_model(model_name):
        if model_name.lower() == "llava":
            return LLaVaModel()
        else:
            raise ValueError("Undefined model name received. Please check the model list to identify the correct model name.")