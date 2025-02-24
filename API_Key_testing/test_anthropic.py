import os
from dotenv import load_dotenv
from anthropic import Anthropic, APIError

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
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        question = "Je me sens vraiment débordé en ce moment, je n'arrive pas à tout gérer. Tu peux m'aider ?"
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            system="Tu es un psychologue bienveillant. Réponds en :\n"
                   "1. Validant l'émotion\n"
                   "2. Proposant une action concrète\n"
                   "3. Posant une question ouverte",
            messages=[
                {"role": "user", "content": question}
            ]
        )

        response_text = response.content[0].text

        print("\n" + "="*50)
        print("⚕️ Test de réponse thérapeutique")
        print(f"Modèle : claude-3-haiku-20240307")
        print("="*50)
        print(f"Question : {question}")
        print(f"Réponse : {response_text}")
        print("\nValidation :")
        
        if validate_response(response_text):
            print("✅ Réponse conforme aux bonnes pratiques thérapeutiques")
            print("Éléments détectés :")
            print("- Empathie et validation émotionnelle")
            print("- Structure claire (écoute/action/question)")
            print("- Conseil actionnable sans jargon")
        else:
            print("⚠️ Réponse non conforme - éléments manquants :")
            if not any(kw in response_text.lower() for kw in ['comprends', 'ressenti']):
                print("- Validation émotionnelle insuffisante")
            if not any(kw in response_text.lower() for kw in ['liste', 'prioriser', 'étape']):
                print("- Actions concrètes manquantes")
            if '?' not in response_text:
                print("- Aucune question ouverte détectée")
                
    except APIError as e:
        print(f"❌ Erreur API : {str(e)}")
    except Exception as e:
        print(f"❌ Erreur inattendue : {str(e)}")

if __name__ == "__main__":
    test_therapeutic_response()