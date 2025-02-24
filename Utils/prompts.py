import tiktoken

__all__ = [
    'basic_qa',
    'cot_reasoning',
    'reflective_listening',
    'solution_focused',
    'deep_self_reflection',
    'cognitive_restructuring',
    'narrative_approach',
    'emotional_first_aid',
    'processcom_adaptative',
    'existential_exploration'
]

def get_max_context_length(model_name: str, question: str) -> int:
    """Calcule dynamiquement la taille maximale du contexte disponible"""
    if 'gpt' in model_name.lower():
        encoder = tiktoken.encoding_for_model(model_name)
        
        # Token limits selon les modèles (mise à jour octobre 2023)
        model_limits = {
            "gpt-4o-mini": 128000,
            "gpt-4o-mini-2024-07-18": 128000,
            "gpt-4o-mini-audio-preview": 128000,
            "gpt-4o-mini-audio-preview-2024-12-17": 128000,
            "gpt-3.5-turbo-0125": 16385
        }
    else:
        # Fallback pour les autres fournisseurs
        return 10000  # Taille fixe générique

    # Calcul du budget token disponible
    question_tokens = len(encoder.encode(question))
    safety_margin = 300  # Pour les instructions système
    max_context = model_limits.get(model_name, 4000) - question_tokens - safety_margin
    
    return max(max_context, 500)  # Garantit un minimum utilisable

def basic_qa(question: str, context: str, model_name: str) -> str:
    """Prompt de base pour réponse thérapeutique structurée"""
    max_len = get_max_context_length(model_name, question)
    truncated_context = context[:max_len]
    
    return f"""En tant que psychologue expérimenté, adopte cette structure :
1. Validation empathique (reformulation + émotion détectée)
2. Hypothèse sur le besoin sous-jacent (lié au contexte)
3. Question ouverte pour exploration
4. Proposition d'action symbolique

Contexte : {truncated_context}
Patient : "{question}"
→ Réponse naturelle (4-5 phrases max) sans jargon technique :"""

def cot_reasoning(question: str, context: str, model_name: str) -> str:
    """Raisonnement en chaîne (Chain-of-Thought)"""
    max_len = get_max_context_length(model_name, question)
    truncated_context = context[:max_len]
    
    return f"""Démarche pas à pas :
1. Identifier l'émotion primaire
2. Détecter les distorsions cognitives
3. Relier au contexte thérapeutique
4. Formuler une hypothèse d'intervention

Contexte : {truncated_context}
Patient : "{question}"
→ Réponse analytique structurée :"""

def reflective_listening(question: str, context: str, model_name: str) -> str:
    """Écoute réflexive avancée"""
    max_len = get_max_context_length(model_name, question)
    truncated_context = context[:max_len]
    
    return f"""Technique en 3 niveaux :
1. Reflet miroir : "Vous dites que [reformulation exacte]..."
2. Écho émotionnel : "Je perçois une nuance de [émotion]..."
3. Profondeur : "Et si on explorait ce qui se cache derrière cela ?"

Contexte : {truncated_context}
Patient : "{question}"
→ Réponse en couches progressives :"""

def deep_self_reflection(question: str, context: str, model_name: str) -> str:
    """Approche réflexive inspirée de DeepThink R1"""
    max_len = get_max_context_length(model_name, question)
    truncated_context = context[:max_len]
    
    return f"""Guide le patient à travers 4 niveaux de réflexion :
[N1 - Surface] "Je perçois que [reformulation]..."
[N2 - Émotion] "Qu'est-ce que cela éveille comme sensations physiques ?"
[N3 - Méta] "Si cette situation était une métaphore, ce serait..."
[N4 - Action] "Quel micro-geste pourrait symboliser un premier pas ?"

Contexte : {truncated_context}
Patient : "{question}"
→ Réponse progressive avec une question par niveau :"""

def cognitive_restructuring(question: str, context: str, model_name: str) -> str:
    """Restructuration cognitive en TCC"""
    max_len = get_max_context_length(model_name, question)
    truncated_context = context[:max_len]
    
    return f"""Applique la méthode ABCDE :
A (Activation) : Identifier la pensée automatique
B (Belief) : Explorer les croyances associées
C (Consequences) : Analyser l'impact émotionnel
D (Dispute) : Recherche de contre-preuves
E (Effective) : Nouvelle perspective équilibrée

Contexte : {truncated_context}
Patient : "{question}"
→ Réponse guidée avec une question à chaque étape :"""

def narrative_approach(question: str, context: str, model_name: str) -> str:
    """Thérapie narrative avec externalisation"""
    max_len = get_max_context_length(model_name, question)
    truncated_context = context[:max_len]
    
    return f"""Utilise ces techniques narratives :
1. Externalisation : "Si ce problème avait un visage, comment le décrirez-vous ?"
2. Cartographie : "Quand ce 'visiteur' se manifeste-t-il le plus ?"
3. Exceptions : "Y a-t-il eu des moments où vous lui avez résisté ?"
4. Ré-autorisation : "Comment renommeriez-vous cette histoire ?"

Contexte : {truncated_context}
Patient : "{question}"
→ Réponse métaphorique et imagée :"""

def emotional_first_aid(question: str, context: str, model_name: str) -> str:
    """Premiers secours émotionnels"""
    max_len = get_max_context_length(model_name, question)
    truncated_context = context[:max_len]
    
    return f"""Procédure en 4 étapes :
1. Ancrage : "Prenez conscience de 3 points de contact avec votre chaise"
2. Respiration : "Inspirez sur 4 temps, bloquez 2, expirez sur 6"
3. Réassurance : "C'est humain de ressentir cela dans cette situation"
4. Ressource : "Quel souvenir réconfortant pourrait vous aider maintenant ?"

Contexte : {truncated_context}
Patient : "{question}"
→ Réponse apaisante avec exercice sensoriel :"""

def processcom_adaptative(question: str, context: str, model_name: str) -> str:
    """Adaptation multi-canaux ProcessCOM"""
    max_len = get_max_context_length(model_name, question)
    truncated_context = context[:max_len]
    
    return f"""Approche multi-sensorielle :
🎨 Visuel : "Si vous deviez dessiner cette émotion..."
🎵 Auditif : "Quelle musique lui associeriez-vous ?"
🤲 Kinesthésique : "Quelle texture représente cet état ?"

Contexte : {truncated_context}
Patient : "{question}"
→ Réponse engageant différents canaux sensoriels :"""

def existential_exploration(question: str, context: str, model_name: str) -> str:
    """Approche existentielle inspirée de Frankl"""
    max_len = get_max_context_length(model_name, question)
    truncated_context = context[:max_len]
    
    return f"""Questions de sens :
1. "Qu'est-ce que cette situation révèle de vos valeurs profondes ?"
2. "Comment pourriez-vous transformer cette épreuve en apprentissage ?"
3. "Quel legs souhaiteriez-vous tirer de cette expérience ?"

Contexte : {truncated_context}
Patient : "{question}"
→ Réponse centrée sur la recherche de sens :"""

def solution_focused(question: str, context: str, model_name: str) -> str:
    """Thérapie orientée solutions avancée"""
    max_len = get_max_context_length(model_name, question)
    truncated_context = context[:max_len]
    
    return f"""Techniques SFA :
🔍 Exception : "Quand le problème est moins présent ?"
🎯 Objectif : "À quoi reconnaîtriez-vous une amélioration ?"
🛠 Ressource : "Quelles compétences avez-vous déjà utilisées ?"
🚀 Échelle : "Sur 10, où êtes-vous ? Que ferait un +1 ?"

Contexte : {truncated_context}
Patient : "{question}"
→ Réponse axée sur les solutions et ressources existantes :"""