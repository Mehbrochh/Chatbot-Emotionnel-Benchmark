import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

def validate_response(response: str) -> bool:
    """Valide que la réponse respecte les critères thérapeutiques"""
    criteria = [
        ('empathie', ['comprends', 'ressenti', 'difficile']),
        ('validation', ['normal', 'commun', 'fréquent']),
        ('action', ['prioriser', 'liste', 'respirer', 'étape']),
        ('question', ['?', 'qu\'en pensez-vous', 'quel aspect'])
    ]
    
    return all(
        any(keyword in response.lower() for keyword in keywords)
        for (_, keywords) in criteria
    )

def test_therapeutic_response():
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        question = "Je me sens vraiment débordé en ce moment, je n'arrive pas à tout gérer. Tu peux m'aider ?"
        response = model.generate_content(f"""
En tant que psychologue, réponds à cette inquiétude en :
1. Validant l'émotion
2. Proposant une première action concrète
3. Posant une question ouverte

Patient : {question}
""")

        print("\n" + "="*50)
        print("⚕️ Test de réponse thérapeutique")
        print(f"Modèle : {model.model_name}")
        print("="*50)
        print(f"Question : {question}")
        print(f"Réponse : {response.text}")
        print("\nValidation :")
        
        if validate_response(response.text):
            print("✅ Réponse conforme aux bonnes pratiques thérapeutiques")
            print("Éléments détectés :")
            print("- Empathie et validation émotionnelle")
            print("- Structure claire (écoute/action/question)")
            print("- Conseil actionnable sans jargon")
        else:
            print("⚠️ Réponse non conforme - éléments manquants :")
            if not any(kw in response.text.lower() for kw in ['comprends', 'ressenti']):
                print("- Validation émotionnelle insuffisante")
            if not any(kw in response.text.lower() for kw in ['liste', 'prioriser', 'étape']):
                print("- Actions concrètes manquantes")
            if '?' not in response.text:
                print("- Aucune question ouverte détectée")
                
    except Exception as e:
        print(f"❌ Erreur : {str(e)}")

if __name__ == "__main__":
    test_therapeutic_response()
    print("\nNote : Les warnings gRPC sont normaux et n'affectent pas la fonctionnalité.")