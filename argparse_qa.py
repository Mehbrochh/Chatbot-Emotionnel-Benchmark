import argparse
import pandas as pd
import time
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import sys
from colorama import Fore, Style

# Importations locales
from Utils import models, prompts

# Configuration des couleurs
COL = {
    'model': Fore.BLUE,
    'prompt': Fore.MAGENTA,
    'question': Fore.CYAN,
    'number': Fore.YELLOW,
    'response': Fore.GREEN,
    'error': Fore.RED,
    'reset': Style.RESET_ALL
}

def print_separator():
    print(f"\n{COL['model']}{'='*40}{COL['reset']}\n")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark de Modèles Thérapeutiques")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["all"],
        help="Modèles à tester (ex: gpt_4o_mini, gpt_3_5_turbo_0125)"
    )
    parser.add_argument(
        "--prompts",
        nargs="*", 
        default=["all"],
        help="Stratégies d'interaction (ex: basic_qa, processcom_prompt)"
    )
    return parser.parse_args()

def load_questions() -> List[str]:
    try:
        path = Path("Utils/Data/questions.txt")
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"{COL['error']}Erreur de chargement des questions: {str(e)}{COL['reset']}")
        sys.exit(1)

def load_processcom() -> str:
    try:
        path = Path("Utils/Data/processcom.txt")
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"{COL['error']}Erreur de chargement du contexte: {str(e)}{COL['reset']}")
        sys.exit(1)

def process_question(model_cfg: Dict, prompt_name: str, question: str, context: str, q_num: int, total_q: int) -> Dict:
    start_time = time.time()
    try:
        prompt_fn = getattr(prompts, prompt_name)
        system_content = prompt_fn(question, context, model_cfg['name'])
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question}
        ]
        
        response = model_cfg['generate_response'](messages)
        execution_time = time.time() - start_time
        
        print_separator()
        print(f"{COL['number']}[Question {q_num}/{total_q}]{COL['reset']}")
        print(f"{COL['model']}Modèle: {model_cfg['name']} | {COL['prompt']}Prompt: {prompt_name}{COL['reset']}")
        print(f"{COL['question']}Patient:{COL['reset']} {question}")
        print(f"{COL['response']}Thérapeute:{COL['reset']} {response.choices[0].message.content}\n")
        print(f"{COL['model']}Stats:{COL['reset']} {execution_time:.2f}s | Tokens: {response.usage.total_tokens}")
        
        return {
            'question': question,
            'response': response.choices[0].message.content,
            'tokens': response.usage.total_tokens,
            'time': execution_time,
            'success': True
        }
    except Exception as e:
        print(f"{COL['error']}Erreur: {str(e)}{COL['reset']}")
        return {
            'question': question,
            'response': f"Error: {str(e)}",
            'tokens': 0,
            'time': time.time() - start_time,
            'success': False
        }

def main():
    args = parse_arguments()
    
    # Chargement des données
    questions = load_questions()
    context = load_processcom()
    
    # Configuration des composants à tester
    model_list = models.__all__ if "all" in args.models else args.models
    prompt_list = prompts.__all__ if "all" in args.prompts else args.prompts
    total_iterations = len(model_list) * len(prompt_list) * len(questions)
    
    results = []
    Path("output").mkdir(exist_ok=True)

    with tqdm(total=total_iterations, desc=f"{COL['model']}Progression{COL['reset']}", unit="req") as pbar:
        for model_name in model_list:
            try:
                model_cfg, _ = getattr(models, model_name)()
            except Exception as e:
                print(f"{COL['error']}Erreur modèle {model_name}: {str(e)}{COL['reset']}")
                continue
            
            for prompt_name in prompt_list:
                if not hasattr(prompts, prompt_name):
                    print(f"{COL['error']}Prompt inconnu: {prompt_name}{COL['reset']}")
                    continue
                
                for idx, question in enumerate(questions, 1):
                    result = process_question(
                        model_cfg=model_cfg,
                        prompt_name=prompt_name,
                        question=question,
                        context=context,
                        q_num=idx,
                        total_q=len(questions)
                    )
                    results.append({
                        'model': model_name,
                        'prompt': prompt_name,
                        **result
                    })
                    
                    if len(results) % 5 == 0:
                        pd.DataFrame(results).to_csv("output/latest.csv", index=False)
                    
                    pbar.update(1)
                    time.sleep(1)  # Gestion des limites de taux

    # Sauvegarde finale
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pd.DataFrame(results).to_csv(f"output/benchmark_{timestamp}.csv", index=False)
    print(f"\n{COL['model']}✅ Benchmark terminé! Fichier: output/benchmark_{timestamp}.csv{COL['reset']}")

if __name__ == "__main__":
    main()