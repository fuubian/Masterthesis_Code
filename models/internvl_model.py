from models.text_image_model import TextImageModel
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

class InternVLModel(TextImageModel):
    def __init__(self, model_name="OpenGVLab/InternVL2_5-8B"):
        super().__init__(model_name)

    def generate_answer(self, question_prompt, image, max_answer_length=500):
        image_object = load_image(image)
        pipe = pipeline(self.model_name, backend_config=TurbomindEngineConfig(session_len=8192), offload_folder="offload_folder")
        model_response = pipe((question_prompt, image_object))

        return model_response.text