import re
import json
from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm
import streamlit as st

def extract_sentences(text: str, word: str) -> List[str]:
    """
    Extrait les phrases contenant un mot spécifique.
    
    Args:
        text (str): Texte source
        word (str): Mot à rechercher
        
    Returns:
        List[str]: Liste des phrases contenant le mot
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if word.lower() in s.lower()]

def get_word_context(text: str, word: str, window: int = 5) -> str:
    """
    Extrait le contexte autour d'un mot avec une fenêtre donnée.
    
    Args:
        text (str): Texte source
        word (str): Mot à rechercher
        window (int): Nombre de mots avant/après
        
    Returns:
        str: Contexte du mot
    """
    words = text.lower().split()
    try:
        idx = words.index(word.lower())
        start = max(0, idx - window)
        end = min(len(words), idx + window + 1)
        return " ".join(words[start:end])
    except ValueError:
        return ""

def find_best_example(text: str, word: str) -> str:
    """
    Trouve le meilleur exemple d'utilisation d'un mot.
    
    Pour le premier extrait de phrase qui respecte les règles suivantes:
    - La phrase doit se terminer par un caractère de ponctuation (".", "!", ou "?").
    - La phrase est convertie en minuscules et nettoyée (suppression des chiffres et caractères spéciaux).
    - Extrait 5 mots avant et 5 mots après le mot cherché (s'il existe).
    
    Args:
        text (str): Texte source
        word (str): Mot à rechercher
        
    Returns:
        str: Contexte extrait si la phrase est valide, sinon une chaîne vide.
    """
    # Séparation en phrases en se basant sur la ponctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        # Vérifier que c'est bien une phrase (se termine par ponctuation)
        if not sentence.strip().endswith(('.', '!', '?')):
            continue
        if word.lower() in sentence.lower():
            # Conversion en minuscules et nettoyage de la phrase:
            cleaned_sentence = sentence.lower()
            # Suppression des chiffres et caractères spéciaux (conserve uniquement les lettres et les espaces)
            cleaned_sentence = re.sub(r'[^a-z\s]', '', cleaned_sentence)
            word_list = cleaned_sentence.split()
            try:
                idx = word_list.index(word.lower())
                start = max(0, idx - 5)
                end = idx + 6  # 5 mots après le mot (le mot lui-même est inclus)
                context = " ".join(word_list[start:end])
                return context
            except ValueError:
                continue
    return ""

def clean_text(text: str) -> str:
    """
    Nettoie le texte en supprimant tous les caractères spéciaux.
    
    Le texte est converti en minuscules et ne conserve que lettres, chiffres, espaces ainsi que la ponctuation essentielle.
    
    Args:
        text (str): Texte à nettoyer
        
    Returns:
        str: Texte nettoyé
    """
    # Conversion en minuscules
    text = text.lower()
    # Suppression de tous les caractères spéciaux (on conserve les lettres, chiffres, espaces et ponctuation simple)
    text = re.sub(r'[^a-z0-9\s.!?]', '', text)
    # Normalisation des espaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_text(text: str, progress_bar=None) -> Dict[str, List[str]]:
    """
    Traite le texte pour extraire les mots uniques et crée un format JSON.
    
    Supprime les mots composés d'une seule lettre.
    
    Args:
        text (str): Texte brut à traiter
        progress_bar: Barre de progression Streamlit (optionnel)
        
    Returns:
        Dict[str, List[str]]: Dictionnaire formaté avec input/output/examples
    """
    # Nettoyage initial du texte (conversion en minuscules et nettoyage simple)
    clean_text_content = clean_text(text)
    
    # Extraction des mots uniques et suppression des mots d'une seule lettre
    words = [word for word in re.findall(r'\b\w+\b', clean_text_content) if word.isalpha() and len(word) > 1]
    unique_words = list(dict.fromkeys(words))
    
    # Création des exemples pour chaque mot
    examples = []
    total = len(unique_words)
    
    if progress_bar:
        progress_text = "Traitement des mots en cours..."
        progress_bar.write(progress_text)
        progress = progress_bar.progress(0)
        for i, word in enumerate(unique_words):
            example = find_best_example(text, word)
            examples.append(example)
            progress.progress((i + 1) / total)
            progress_text = f"Traitement: {i+1}/{total} mots"
            progress_bar.write(progress_text)
    else:
        for word in tqdm(unique_words, desc="Extraction des exemples"):
            example = find_best_example(text, word)
            examples.append(example)
    
    result = {
        "input": unique_words,
        "output": [""] * len(unique_words),
        "examples": examples
    }
    
    return result

def save_to_json(data: Dict, output_file: str):
    """Sauvegarde les données dans un fichier JSON."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Version console
if __name__ == "__main__":
    try:
        with open('data/clean/datasets.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        print("🔄 Début du traitement...")
        result = process_text(text)
        print("💾 Sauvegarde des résultats...")
        save_to_json(result, 'data/processed/output.json')
        print("✅ Traitement terminé avec succès!")
        print(f"📊 Statistiques:")
        print(f"- Mots uniques: {len(result['input'])}")
        print(f"- Exemples trouvés: {len([x for x in result['examples'] if x])}")
    except Exception as e:
        print(f"❌ Erreur lors du traitement: {str(e)}")