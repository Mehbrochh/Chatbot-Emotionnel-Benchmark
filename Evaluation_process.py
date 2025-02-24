import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import time
from datetime import datetime
from pathlib import Path
from Utils.context_manager import ContextManager

# Charger les variables d'environnement et initialiser les composants
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
context_manager = ContextManager()

def get_relevant_context(question):
    """Récupère le contexte pertinent des deux sources."""
    processcom_context = context_manager.get_context("processcom", question)
    emotion_study_context = context_manager.get_context("emotion_study", question)
    return f"{processcom_context}\n\n{emotion_study_context}"

def evaluate_response(question, response):
    """Évalue la qualité d'une réponse en utilisant GPT-4 avec le contexte RAG."""
    try:
        context = get_relevant_context(question)
        
        prompt = f"""En tant que psychologue clinicien expert, évaluez cette réponse selon la structure attendue et le contexte fourni.

        Contexte théorique pertinent:
        {context}

        Structure de réponse idéale (notation stricte):
        1. Commencer par l'empathie
        2. Poser des questions exploratoires
        3. Fournir des explications/conseils
        4. Ne jamais interrompre le flux de pensée du patient

        Structure pénalisée (-3 points minimum):
        - Empathie suivie directement d'hypothèses sur l'origine
        - Questions posées après avoir donné des explications
        - Interruption du discours du patient

        Critères d'évaluation:
        - Respect strict de la structure de réponse (40% de la note)
        - Alignement avec ProcessCom et l'étude des émotions (30%)
        - Qualité de l'empathie et exploration (30%)

        Échelle:
        1-3 : Structure incorrecte ou réponse inadéquate
        4-5 : Structure partiellement respectée, contenu basique
        6-7 : Bonne structure mais application partielle
        8-9 : Excellente structure et contenu
        10 : Structure parfaite et contenu exceptionnel

        Question : {question}
        Réponse à évaluer : {response}

        Donnez uniquement une note numérique sur 10 (avec décimales possibles)."""

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """Vous êtes un psychologue clinicien exigeant qui insiste particulièrement sur la structure 
                empathie → questions → explications. Toute déviation de cette structure doit être sévèrement sanctionnée. 
                Répondez uniquement avec un nombre."""},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        
        note = float(completion.choices[0].message.content.strip())
        return note
        
    except Exception as e:
        print(f"Erreur lors de l'évaluation : {e}")
        return None

def is_success(value):
    """Vérifie si la valeur correspond à un succès dans différents formats possibles."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ['true', 'oui', 'yes', '1']
    return False

def main():
    output_dir = Path('output_evaluated')
    output_dir.mkdir(exist_ok=True)
    
    input_path = 'output/benchmark_20250121-GPT-Turbo.csv'
    try:
        df = pd.read_csv(input_path, sep=',')
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier CSV : {e}")
        return

    print("Colonnes trouvées dans le CSV :", df.columns.tolist())
    
    df['Note_Qualité'] = None
    
    # Identifier les colonnes avec gestion de casse flexible
    question_col = next((col for col in df.columns if col.lower() in ['question', 'requête']), None)
    response_col = next((col for col in df.columns if col.lower() in ['response', 'réponse']), None)
    success_col = next((col for col in df.columns if col.lower() in ['success', 'succès']), None)

    if not all([question_col, response_col, success_col]):
        print("Colonnes requises non trouvées dans le CSV")
        return
    
    total_notes = 0
    count_notes = 0
    
    for index, row in df.iterrows():
        if is_success(row[success_col]):
            print(f"\nÉvaluation de la ligne {index + 1}...")
            try:
                note = evaluate_response(row[question_col], row[response_col])
                if note is not None:
                    df.at[index, 'Note_Qualité'] = note
                    total_notes += note
                    count_notes += 1
                    print(f"Question : {row[question_col][:100]}...")
                    print(f"Réponse : {row[response_col][:100]}...")
                    print(f"Note : {note}")
                    print(f"Moyenne actuelle : {total_notes/count_notes:.2f}")
                time.sleep(1)
            except Exception as e:
                print(f"Erreur lors du traitement de la ligne {index + 1}: {e}")
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f'benchmark_evaluated_{timestamp}.csv'
    
    try:
        df.to_csv(output_path, index=False)
        print(f"\nÉvaluation terminée. Résultats sauvegardés dans {output_path}")
        if count_notes > 0:
            print(f"Note moyenne finale : {total_notes/count_notes:.2f}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du fichier : {e}")

if __name__ == "__main__":
    main()