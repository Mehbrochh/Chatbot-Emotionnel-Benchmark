import tiktoken

__all__ = [
    'basic_qa', 
    'cot_reasoning', 
    'reflective_listening',
    'solution_focused',
    'processcom_prompt'
]

def get_max_context_length(model_name: str, question: str) -> int:
    """Calcule dynamiquement la taille maximale du contexte"""
    encoder = tiktoken.encoding_for_model(model_name)
    
    # Token limits selon les modèles (à mettre à jour selon la documentation officielle)
    model_limits = {
        "gpt-4o-mini": 128000,
        "gpt-4o-mini-2024-07-18": 128000,
        "gpt-4o-mini-audio-preview": 128000,
        "gpt-4o-mini-audio-preview-2024-12-17": 128000,
        "gpt-3.5-turbo-0125": 16385
    }
    
    # Budget tokens: limite - (question + instructions + marge de sécurité)
    question_tokens = len(encoder.encode(question))
    instructions_tokens = 500  # Estimation moyenne des instructions du prompt
    safety_margin = 200
    
    max_length = model_limits.get(model_name, 4000) - question_tokens - instructions_tokens - safety_margin
    
    return max(max_length, 0)  # Garantit une valeur positive

def basic_qa(question: str, context: str, model_name: str) -> str:
    max_len = get_max_context_length(model_name, question)
    truncated_context = context[:max_len]
    
    return f"""En tant que thérapeute, engagez la conversation :
1. Validation émotionnelle
2. Lien avec un besoin sous-jacent
3. Proposition d'action concrète
4. Question ouverte

Contexte : {truncated_context}
Patient : "{question}"
→ Réponse (4 phrases max) :"""

def cot_reasoning(question: str, context: str, model_name: str) -> str:
    max_len = get_max_context_length(model_name, question)
    truncated_context = context[:max_len]
    
    return f"""Analyse en 3 phases :
1. Mise en perspective
2. Hypothèse thérapeutique
3. Pont vers l'action

Contexte : {truncated_context}
Patient : "{question}"
→ Réponse structurée :"""

def reflective_listening(question: str, context: str, model_name: str) -> str:
    max_len = get_max_context_length(model_name, question)
    truncated_context = context[:max_len]
    
    return f"""Technique d'écoute active :
1. Reflet émotionnel
2. Validation
3. Ouverture stratégique

Contexte : {truncated_context}
Patient : "{question}"
→ Réponse réflexive :"""

def solution_focused(question: str, context: str, model_name: str) -> str:
    max_len = get_max_context_length(model_name, question)
    truncated_context = context[:max_len]
    
    return f"""Approche orientée solutions :
1. Exception positive
2. Ressource identifiée
3. Projection future
4. Étape symbolique

Contexte : {truncated_context}
Patient : "{question}"
→ Réponse actionnable :"""

def processcom_prompt(question: str, context: str, model_name: str) -> str:
    max_len = get_max_context_length(model_name, question)
    truncated_context = context[:max_len]
    
    return f"""Adaptation multi-canaux :
- Visuel 🎨
- Auditif 🎵 
- Kinesthésique 🤲

Contexte : {truncated_context}
Patient : "{question}"
→ Réponse adaptée :"""