# Chatbot-Emotionnel-Benchmark
### < 🤖 Assistant Thérapeutique - Benchmark IA >


## 🚀 Démarrage Rapide

### Prérequis
```bash
Python >= 3.10

### **🧠 Utilisation**
**-> _Comment exécuter le code ?_**

\\ Lancer tous les modèles/prompts
python argparse_qa.py

\\ Tester des combinaisons spécifiques
python argparse_qa.py --models gpt-4o-mini --prompts basic_qa

Options disponibles

--models :
    __all__ = [
        'gpt-4o-mini',
        'gpt-4o-mini-2024-07-18',
        'gpt-4o-mini-audio-preview',
        'gpt-4o-mini-audio-preview-2024-12-17',
        'gpt-3.5-turbo-0125'
    ]

--prompts	:
    __all__ = [
      'basic_qa', 
      'cot_reasoning', 
      'reflective_listening',
      'solution_focused',
      'processcom_prompt'
    ]

### 🛠 Personnalisation
1. Améliorer les scénarios

-> Éditez **Utils/Data/questions.txt** :

\\ Format : Une question/par ligne !
    Je lutte contre une anxiété chronique...
    Mes relations professionnelles sont tendues...
    J'ai perdu motivation pour mes hobbies...

2. Ajouter un nouveau prompt

a -> Éditez **Utils/prompts.py** :

    def mon_prompt_perso(question: str, context: str) -> str:
        return f"""**Ma stratégie**
    1. Identifier l'émotion dominante : {question}
    2. Analyser : {context[:200]}
    3. Proposer une action concrète
    """

b -> Ajoutez-le à la liste **__all__** :

    __all__ = [..., 'mon_prompt_perso']  # 👈 N'oubliez pas cette ligne !

c -> Testez :

    python argparse_qa.py --prompts mon_prompt_perso

### 📂 Structure du Code

      .
      ├── .env                    # Configuration
      ├── argparse_qa.py          # Script principal
      ├── Utils/
      │   ├── models.py           # Configurations des IA
      │   ├── prompts.py          # Stratégies (à modifier)
      │   └── Data/
      │       └── questions.txt   # Scénarios (à enrichir)
      │       └── PCM.txt
      └── output/                 # Résultats des tests

### 📈 Amélioration Continue

-> Fichiers clés à modifier :

    **questions.txt** : Ajoutez ou modifiez avec des cas complexes

    **prompts.py** : Développez de nouvelles stratégies

    **models.py** : Intégrez de nouveaux modèles (à moi de gérer)

### Workflow recommandé :

    Modifiez questions.txt avec vos scénarios

    Créez un nouveau prompt dans prompts.py

    Testez avec :

        python argparse_qa.py --models gpt-4o-mini --prompts votre_prompt

N'hésitez pas à me contacter en cas de besoin !
