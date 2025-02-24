import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
import pandas as pd
import time
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut
import sys
from colorama import Fore, Style
import tiktoken

from Utils import models, prompts
from Utils.context_manager import ContextManager

COL = {
    'model': Fore.BLUE,
    'prompt': Fore.MAGENTA,
    'question': Fore.CYAN,
    'number': Fore.YELLOW,
    'response': Fore.GREEN,
    'error': Fore.RED,
    'stats': Fore.WHITE,
    'context': Fore.LIGHTBLUE_EX,
    'reset': Style.RESET_ALL
}

def print_separator():
    print(f"\n{COL['model']}{'='*80}{COL['reset']}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark Thérapeutique avec RAG")
    parser.add_argument("--providers", nargs="+", default=["all"], 
                      help="Fournisseurs à tester : openai, google, anthropic")
    parser.add_argument("--models", nargs="+", default=["all"], 
                      help=f"Modèles disponibles: {', '.join(models.__all__)}")
    parser.add_argument("--prompts", nargs="+", default=["all"], 
                      help=f"Prompts disponibles: {', '.join(prompts.__all__)}")
    parser.add_argument("--contexts", nargs="+", default=["all"], 
                      help="Contextes disponibles: processcom, emotion_study")
    return parser.parse_args()

def load_questions() -> List[str]:
    path = Path("Utils/Data/questions.txt")
    try:
        return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    except Exception as e:
        sys.exit(f"{COL['error']}Erreur de chargement des questions: {str(e)}{COL['reset']}")

def process_question(model_cfg: Dict, prompt_name: str, question: str, 
                    context: str, context_source: str, q_num: int, total_q: int) -> Dict:
    start_time = time.time()
    full_response = ""
    
    try:
        prompt_fn = getattr(prompts, prompt_name)
        system_content = prompt_fn(question, context, model_cfg['name'])
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question}
        ]

        print_separator()
        print(f"{COL['number']}[Question {q_num}/{total_q}]{COL['reset']}")
        print(f"{COL['context']}Contexte: {context_source}{COL['reset']} | "
              f"{COL['model']}Fournisseur: {model_cfg['provider']}{COL['reset']} | "
              f"{COL['prompt']}Modèle: {model_cfg['name']}{COL['reset']} | "
              f"{COL['prompt']}Prompt: {prompt_name}{COL['reset']}")
        print(f"{COL['question']}Patient:{COL['reset']}\n{question}")
        print(f"{COL['response']}Thérapeute:{COL['reset']} ", end="", flush=True)

        response_stream = model_cfg['generate_response'](messages, stream=True)
        
        # Gestion multi-fournisseur du streaming
        for chunk in response_stream:
            content = ""
            if model_cfg['provider'] == 'openai':
                content = chunk.choices[0].delta.content or ""
            elif model_cfg['provider'] == 'google':
                content = chunk.text if hasattr(chunk, 'text') else ""
            elif model_cfg['provider'] == 'anthropic':
                content = chunk.content[0].text if hasattr(chunk, 'content') else ""
            
            if content:
                print(f"{COL['response']}{content}{COL['reset']}", end="", flush=True)
                full_response += content

        execution_time = time.time() - start_time
        
        # Calcul des tokens selon le fournisseur
        token_count = len(full_response.split())  # Fallback générique
        if model_cfg['provider'] == 'openai':
            encoder = tiktoken.encoding_for_model(model_cfg['name'])
            token_count = len(encoder.encode(full_response))
        else:
            # Estimation pour les autres fournisseurs
            token_count = len(full_response.split())  # 1 token ≈ 1 mot
        
        print(f"\n\n{COL['stats']}――――――――――――――――――――――――――――――――――――――――――――――――――――――――")
        print(f"⏱ Temps: {execution_time:.2f}s | 📊 Tokens: {token_count}")
        print(f"――――――――――――――――――――――――――――――――――――――――――――――――――――――――{COL['reset']}\n")

        return {
            'response': full_response,
            'tokens': token_count,
            'time': execution_time,
            'success': True
        }

    except Exception as e:
        print(f"{COL['error']}\nErreur: {str(e)}{COL['reset']}")
        return {
            'response': f"Error: {str(e)}",
            'tokens': 0,
            'time': time.time() - start_time,
            'success': False
        }
def main():
    args = parse_arguments()
    context_manager = ContextManager()
    
    # Configuration dynamique des fournisseurs
    providers = args.providers if "all" not in args.providers else ['openai', 'google', 'anthropic']
    
    # Configuration des modèles
    model_list = models.__all__ if "all" in args.models else [m for m in models.__all__ if m in args.models]
    
    # Gestion des contextes
    contexts_to_process = []
    for ctx in args.contexts:
        if ctx == "all":
            contexts_to_process.extend(["processcom", "emotion_study"])
        else:
            contexts_to_process.append(ctx)
    
    # Gestion des prompts
    prompts_to_process = prompts.__all__ if "all" in args.prompts else [p for p in prompts.__all__ if p in args.prompts]
    
    # Chargement des questions
    questions = load_questions()
    
    # Calcul du nombre total de requêtes
    total = len(model_list) * len(prompts_to_process) * len(contexts_to_process) * len(questions)
    
    results = []
    Path("output").mkdir(exist_ok=True)

    with tqdm(total=total, desc=f"{COL['model']}Progression globale{COL['reset']}", unit="req") as pbar:
        for context in contexts_to_process:
            try:
                ctx_manager = context_manager.get_context(context, "")  # Initialisation du contexte
            except Exception as e:
                print(f"{COL['error']}Erreur de contexte {context}: {str(e)}{COL['reset']}")
                continue

            for model_name in model_list:
                try:
                    model_cfg, _ = getattr(models, model_name)()
                    if model_cfg['provider'].lower() not in [p.lower() for p in providers]:
                        continue

                    for prompt_name in prompts_to_process:
                        if not hasattr(prompts, prompt_name):
                            print(f"{COL['error']}Prompt inconnu: {prompt_name}{COL['reset']}")
                            continue

                        for q_num, question in enumerate(questions, 1):
                            try:
                                ctx = context_manager.get_context(context, question)
                                result = process_question(
                                    model_cfg=model_cfg,
                                    prompt_name=prompt_name,
                                    question=question,
                                    context=ctx,
                                    context_source=context,
                                    q_num=q_num,
                                    total_q=len(questions)
                                )

                                results.append({
                                    "Fournisseur": model_cfg['provider'],
                                    "Modèle": model_cfg['name'],
                                    "Prompt": prompt_name,
                                    "Contexte": context,
                                    "Question": question,
                                    "Réponse": result['response'],
                                    "Tokens": result['tokens'],
                                    "Temps": round(result['time'], 2),
                                    "Succès": "Oui" if result['success'] else "Non"
                                })

                            except Exception as e:
                                print(f"{COL['error']}Erreur question {q_num}: {str(e)}{COL['reset']}")
                            
                            pbar.update(1)
                            time.sleep(0.2)  # Rate limiting

                except Exception as e:
                    print(f"{COL['error']}Erreur modèle {model_name}: {str(e)}{COL['reset']}")
                    continue

    # Sauvegarde des résultats
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    df = pd.DataFrame(results)
    df.to_csv(f"output/benchmark_{timestamp}.csv", index=False, encoding='utf-8-sig')
    print(f"\n{COL['model']}✅ Benchmark terminé! Fichier: output/benchmark_{timestamp}.csv{COL['reset']}")

if __name__ == "__main__":
    main()