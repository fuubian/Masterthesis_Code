import sys
import random
from utils.text_image_loader import TextImageLoader
from utils.data_loader import DataLoader

def main(model_name):
    """
    This function tests if inference for a given model is applicable. It can be used before running a model on the whole dataset.

    Args:
        model_name (str): The name of the model. Please consult the model list to find the correct term.
    """
    # Trying to load the model
    model = None
    try:
        model = TextImageLoader.load_model(model_name)
        print(f"{model_name} was successfully initialized.")
    except Exception as e:
        print(f"{model_name} could not be initialized: {e}")
        sys.exit()

    # Trying to run inference for one qa_pair
    data = DataLoader.load_task_data(2)
    test_prompt = "Explain me what you can see in this image."
    image = random.choice(list(data.keys()))
    try:
        response = model.generate_answer(test_prompt, image)
        print(f"Model response: {response}")
    except Exception as e:
        print(f"{model_name} was not able to produce a response: {e}")

if __name__ == '__main__':
    NUMBER_OF_ARGUMENTS = 1
    args = sys.argv[1:]
    if len(args) != NUMBER_OF_ARGUMENTS:
        print(f"Unexpeceted number of arguments received. Expected: {NUMBER_OF_ARGUMENTS}; Received: {len(args)}")
    else:
        try:
            model_name = args[0]
        except ValueError as e:
            print(f"Error occurred while processing arguments: {e}")
        else:
            main(model_name)