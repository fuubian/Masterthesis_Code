import os
from dotenv import load_dotenv

class TokenLoader:
    @staticmethod
    def load_token_huggingface():
        """This function returns the token stored in .env to access hugging face models."""
        load_dotenv()
        access_token = os.getenv("ACCESS_TOKEN_HUGGINGFACE")
        return access_token
    
    @staticmethod
    def load_api_key_openai():
        load_dotenv()
        api_key = os.getenv("API_KEY_OPENAI")
        return api_key