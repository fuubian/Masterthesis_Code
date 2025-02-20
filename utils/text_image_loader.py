import config
from models import llava2_model, instructblip_model, internvl_model, ovis_model, paligemma_model, pixtral_model, qwen_model
#from models import deepseek_model

"""
IMPORTANT: To run deepseek, the github repository must be available in the folder.
If it is, uncomment the import line upwards and the return line downwards.
"""

class TextImageLoader:
    @staticmethod
    def load_model(model_name):
        if model_name.lower() == config.DEEPSEEK_MODEL_NAME:
            #return deepseek_model.DeepSeekModel()
            return None
        if model_name.lower() == config.INSTRCUCTBLIP_MODEL_NAME:
            return instructblip_model.InstructBlipModel()
        if model_name.lower() == config.INTERNVL_MODEL_NAME:
            return internvl_model.InternVLModel()
        if model_name.lower() == config.LLAVA_MODEL_NAME:
            return llava2_model.LLaVaModel()
        if model_name.lower() == config.OVIS_MODEL_NAME:
            return ovis_model.OvisModel()
        if model_name.lower() == config.PALIGEMMA_MODEL_NAME:
            return paligemma_model.PaligemmaModel()
        if model_name.lower() == config.PIXTRAL_MODEL_NAME:
            return pixtral_model.PixtralModel()
        if model_name.lower() == config.QWEN_MODEL_NAME:
            return qwen_model.QwenModel()
        raise ValueError("Undefined model name received. Please check the model list to identify the correct model name.")