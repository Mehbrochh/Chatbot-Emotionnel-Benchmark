from openai import OpenAI, APIError, APIConnectionError
import os
from dotenv import load_dotenv
from typing import Tuple, Dict
import time
import logging

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

__all__ = [
    'gpt_4o_mini',
    'gpt_4o_mini_2024_07_18',
    'gpt_4o_mini_audio_preview',
    'gpt_4o_mini_audio_preview_2024_12_17',
    'gpt_3_5_turbo_0125'
]

def create_model(
    model_name: str,
    max_tokens: int = 2000,
    temperature: float = 0.7,
    timeout: int = 45
) -> Tuple[Dict, None]:
    
    def generate_response(messages: list, timeout: int = timeout):
        retry_count = 0
        backoff_factor = 1
        
        while retry_count <= 3:
            try:
                return client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout
                )
            except APIConnectionError:
                time.sleep(backoff_factor * 2)
                backoff_factor *= 2
                retry_count += 1
            except APIError as e:
                logging.error(f"API Error: {e.status_code} {e.message}")
                raise
        
        raise RuntimeError("Échec après 3 tentatives")

    return {
        'name': model_name,
        'generate_response': generate_response,
        'config': {
            'max_tokens': max_tokens,
            'temperature': temperature,
            'timeout': timeout
        }
    }, None

# Configurations spécifiques
def gpt_4o_mini() -> Tuple[Dict, None]:
    return create_model(
        model_name="gpt-4o-mini",
        max_tokens=4000,
        timeout=60
    )

def gpt_4o_mini_2024_07_18() -> Tuple[Dict, None]:
    return create_model(
        model_name="gpt-4o-mini-2024-07-18",
        max_tokens=3500
    )

def gpt_4o_mini_audio_preview() -> Tuple[Dict, None]:
    return create_model(
        model_name="gpt-4o-mini-audio-preview",
        max_tokens=1000,
        temperature=0.5,
        timeout=90
    )

def gpt_4o_mini_audio_preview_2024_12_17() -> Tuple[Dict, None]:
    return create_model(
        model_name="gpt-4o-mini-audio-preview-2024-12-17",
        max_tokens=1500
    )

def gpt_3_5_turbo_0125() -> Tuple[Dict, None]:
    return create_model(
        model_name="gpt-3.5-turbo-0125",
        max_tokens=3000
    )