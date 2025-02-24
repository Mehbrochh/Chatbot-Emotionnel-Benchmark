import os
import time
import logging
from typing import Tuple, Dict, Any
from dotenv import load_dotenv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Désactive les logs TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Désactive complètement CUDA

# Import des SDKs multi-fournisseurs
from openai import OpenAI, APIError, APIConnectionError
import google.generativeai as genai
from anthropic import Anthropic

load_dotenv()

class ModelFactory:
    """Factory pour gérer plusieurs fournisseurs de modèles"""
    
    _providers = {
        'openai': OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        'google': genai.configure(api_key=os.getenv("GOOGLE_API_KEY")),
        'anthropic': Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
    }


    @classmethod
    def create_model(cls, provider: str, model_name: str, **kwargs) -> Tuple[Dict[str, Any], None]:
        """Crée une configuration de modèle pour un fournisseur spécifique"""
        config = {
            'max_tokens': 2000,
            'temperature': 0.7,
            'timeout': 45,
            **kwargs
        }
        
        def generate_response(messages: list, stream: bool = False, timeout: int = config['timeout']):
            retry_count = 0
            backoff_factor = 1
            
            while retry_count <= 3:
                try:
                    if provider == 'openai':
                        return cls._providers['openai'].chat.completions.create(
                            model=model_name,
                            messages=messages,
                            stream=stream,
                            **{k:v for k,v in config.items() if k != 'timeout'}
                        )
                    # Dans models.py - Section Google modifiée avec logging

                    elif provider == 'google':
                        def generate_response(messages: list, stream: bool = False, timeout: int = config['timeout']):
                            try:
                                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                                model = genai.GenerativeModel(model_name)
                                
                                # Combine system and user messages
                                combined_prompt = ""
                                for message in messages:
                                    if message['role'] == 'system':
                                        combined_prompt += message['content'] + "\n"
                                    elif message['role'] == 'user':
                                        combined_prompt += message['content']

                                safety_settings = {
                                    "HARASSMENT": "block_none",
                                    "HATE_SPEECH": "block_none",
                                    "SEXUALLY_EXPLICIT": "block_none",
                                    "DANGEROUS_CONTENT": "block_none",
                                }

                                generation_config = {
                                    "temperature": config['temperature'],
                                    "top_p": 0.8,
                                    "top_k": 40,
                                    "max_output_tokens": config['max_tokens'],
                                }

                                # Handle streaming
                                if stream:
                                    try:
                                        response = model.generate_content(
                                            combined_prompt,
                                            generation_config=generation_config,
                                            safety_settings=safety_settings,
                                            stream=True
                                        )
                                        
                                        class StreamWrapper:
                                            def __iter__(self):
                                                for chunk in response:
                                                    if hasattr(chunk, 'text') and chunk.text:
                                                        yield type('Response', (), {
                                                            'text': chunk.text
                                                        })
                                        
                                        return StreamWrapper()
                                    except Exception as e:
                                        # En cas d'erreur de streaming, on tente une réponse non-streamée
                                        response = model.generate_content(
                                            combined_prompt,
                                            generation_config=generation_config,
                                            safety_settings=safety_settings,
                                            stream=False
                                        )
                                        class SingleResponseWrapper:
                                            def __iter__(self):
                                                yield type('Response', (), {
                                                    'text': response.text
                                                })
                                        return SingleResponseWrapper()
                                else:
                                    response = model.generate_content(
                                        combined_prompt,
                                        generation_config=generation_config,
                                        safety_settings=safety_settings
                                    )
                                    return response

                            except Exception as e:
                                logging.error(f"Erreur Gemini: {str(e)}")
                                raise Exception(f"Erreur Gemini: {str(e)}")
        
                    elif provider == 'anthropic':
                        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                        response = client.messages.create(
                            model=model_name,
                            messages=messages,
                            max_tokens=config['max_tokens'],
                            temperature=config['temperature'],
                            stream=stream
                        )
                        
                        # Gestion spécifique du streaming
                        if stream:
                            class AnthropicStream:
                                def __iter__(self):
                                    for chunk in response:
                                        yield type('obj', (object,), {'content': [type('obj', (object,), {'text': chunk.delta.content})]})
                            return AnthropicStream()
                        return response
                        
                except (APIConnectionError, genai.core.exceptions.APIError) as e:
                    logging.error(f"Erreur de connexion: {str(e)}")
                    time.sleep(backoff_factor * 2)
                    backoff_factor *= 2
                    retry_count += 1
                    
            raise RuntimeError("Échec après 3 tentatives")

        return {
            'name': model_name,
            'generate_response': generate_response,
            'config': config,
            'provider': provider
        }, None

# Configuration des modèles existants

# OpenAI
def gpt_4o_mini() -> Tuple[Dict, None]:
    return ModelFactory.create_model(
        provider='openai',
        model_name="gpt-4o-mini",
        max_tokens=4000,
        timeout=60
    )

def gpt_4o_mini_2024_07_18() -> Tuple[Dict, None]:
    return ModelFactory.create_model(
        provider='openai',
        model_name="gpt_4o_mini_2024_07_18",
        max_tokens=3500,
        timeout=60
    )
    
def gpt_3_5_turbo_0125() -> Tuple[Dict, None]:
    return ModelFactory.create_model(
        provider='openai',
        model_name="gpt_3_5_turbo_0125",
        max_tokens=3000,
        timeout=60
    )
    
# Google AI Studio
def gemini_flash() -> Tuple[Dict, None]:
    return ModelFactory.create_model(
        provider='google',
        model_name="gemini-1.5-flash", 
        max_tokens=2048,
        temperature=0.7,
        timeout=60
    )


# Anthropic (Claude)
def claude_3_haiku() -> Tuple[Dict, None]:
    return ModelFactory.create_model(
        provider='anthropic',
        model_name="claude-3-haiku@20240307",
        max_tokens=4000
    )


__all__ = [
    # OpenAI
    'gpt_4o_mini',
    'gpt_4o_mini_2024_07_18',
    'gpt_3_5_turbo_0125',
    
    # Nouveaux fournisseurs
    'gemini_flash',
    'claude_3_haiku'
]