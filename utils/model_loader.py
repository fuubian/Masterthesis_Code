import config
from models import instructblip_model, internvl_model, gpt4_model, llava_model, ovis_model, paligemma_model, qwen_model, glm_model, minicpm_model

class ModelLoader:
    @staticmethod
    def load_model(model_name):
        if model_name.lower() == config.GLM_MODEL_NAME:
            return glm_model.GLMModel()
        if model_name.lower() == config.GPT_MODEL_NAME:
            return gpt4_model.GPTModel()
        if model_name.lower() == config.INSTRCUCTBLIP_MODEL_NAME:
            return instructblip_model.InstructBlipModel()
        if model_name.lower() == config.INTERNVL_MODEL_NAME:
            return internvl_model.InternVLModel()
        if model_name.lower() == config.INTERNVL_MODEL_NAME_1B:
            return internvl_model.InternVLModel(model_name="OpenGVLab/InternVL2_5-1B")
        if model_name.lower() == config.INTERNVL_MODEL_NAME_2B:
            return internvl_model.InternVLModel(model_name="OpenGVLab/InternVL2_5-2B")
        if model_name.lower() == config.INTERNVL_MODEL_NAME_4B:
            return internvl_model.InternVLModel(model_name="OpenGVLab/InternVL2_5-4B")
        if model_name.lower() == config.LLAVA_MODEL_NAME:
            return llava_model.LLaVaModel()
        if model_name.lower() == config.MINICPM_MODEL_NAME:
            return minicpm_model.MiniCPMModel()
        if model_name.lower() == config.OVIS_MODEL_NAME:
            return ovis_model.OvisModel()
        if model_name.lower() == config.PALIGEMMA_MODEL_NAME:
            return paligemma_model.PaligemmaModel()
        if model_name.lower() == config.PALIGEMMA_MODEL_NAME_3B:
            return paligemma_model.PaligemmaModel(model_name="google/paligemma2-3b-pt-448")
        if model_name.lower() == config.PALIGEMMA_MODE_NAME_FINETUNED:
            return paligemma_model.PaligemmaModel(model_name="fuubian/trained_paligemma")
        if model_name.lower() == config.QWEN_MODEL_NAME:
            return qwen_model.QwenModel()
        raise ValueError(f"Undefined model name received: {model_name}. Please check the model list to identify the correct model name.")