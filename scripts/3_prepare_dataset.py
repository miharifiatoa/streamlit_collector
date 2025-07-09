import pandas as pd
import json
import os
import argparse
import random
from tqdm import tqdm
import time
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def create_dataset(input_file, output_file, window_size=1, test_split=0.1, validate=True, seed=42):
    """
    Crée un dataset au format JSONL pour l'entraînement de modèles linguistiques.
    
    Args:
        input_file: Chemin vers le fichier de texte nettoyé
        output_file: Chemin vers le fichier de sortie JSONL
        window_size: Nombre de lignes à inclure comme contexte pour l'entrée
        test_split: Proportion des données à réserver pour le test (0-1)
        validate: Si True, effectue une validation des données générées
        seed: Graine pour la reproductibilité
    """
    start_time = time.time()
    
    # Vérification de l'existence du fichier d'entrée
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier d'entrée introuvable: {input_file}")
    
    # Lecture du fichier d'entrée
    logger.info(f"Lecture du fichier {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    logger.info(f"Création du dataset avec {len(lines)} lignes de texte...")
    
    # Préparation des données
    data = []
    
    # Utilisation de tqdm pour afficher une barre de progression
    for i in tqdm(range(len(lines) - window_size)):
        # Construction du contexte avec window_size lignes précédentes
        if window_size > 1:
            input_text = " ".join(lines[i:i+window_size])
        else:
            input_text = lines[i]
        
        output_text = lines[i + window_size]
        
        # Vérification de la qualité des données
        if len(input_text) < 5 or len(output_text) < 5:
            continue  # Ignorer les entrées/sorties trop courtes
        
        data.append({
            "input": input_text,
            "output": output_text
        })
    
    # Mélange des données pour la répartition train/test
    random.seed(seed)
    random.shuffle(data)
    
    # Division en ensembles d'entraînement et de test
    split_idx = int(len(data) * (1 - test_split))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # Création des chemins de sortie
    output_path = Path(output_file)
    train_path = output_path.with_name(f"{output_path.stem}_train{output_path.suffix}")
    test_path = output_path.with_name(f"{output_path.stem}_test{output_path.suffix}")
    
    # Création du répertoire de sortie si nécessaire
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Écriture des fichiers JSONL
    logger.info(f"Écriture de {len(train_data)} exemples d'entraînement...")
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Écriture de {len(test_data)} exemples de test...")
    with open(test_path, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Statistiques des données
    input_lengths = [len(item["input"]) for item in data]
    output_lengths = [len(item["output"]) for item in data]
    
    # Validation du dataset (si demandé)
    if validate:
        logger.info("Validation du dataset...")
        # Test de lecture du fichier JSONL généré
        try:
            with open(train_path, 'r', encoding='utf-8') as f:
                first_lines = [json.loads(next(f)) for _ in range(min(5, len(train_data)))]
            logger.info(f"Validation réussie. Exemple d'entrée: {first_lines[0]}")
        except Exception as e:
            logger.error(f"Erreur lors de la validation: {e}")
    
    processing_time = time.time() - start_time
    
    # Rapport final
    logger.info("✅ Création du dataset terminée")
    logger.info(f"⏱️ Temps de traitement: {processing_time:.2f} secondes")
    logger.info(f"📊 Statistiques:")
    logger.info(f"  - Total: {len(data)} paires")
    logger.info(f"  - Entraînement: {len(train_data)} exemples → {train_path}")
    logger.info(f"  - Test: {len(test_data)} exemples → {test_path}")
    logger.info(f"  - Longueur moyenne entrée: {sum(input_lengths)/len(input_lengths):.1f} caractères")
    logger.info(f"  - Longueur moyenne sortie: {sum(output_lengths)/len(output_lengths):.1f} caractères")
    
    return {
        "train_file": str(train_path),
        "test_file": str(test_path),
        "train_count": len(train_data),
        "test_count": len(test_data),
        "total_count": len(data)
    }

if __name__ == "__main__":
    # Configuration du parser d'arguments
    parser = argparse.ArgumentParser(description="Crée un dataset au format JSONL pour l'entraînement de modèles linguistiques")
    parser.add_argument("--input", default="data/clean/clean_texte_antandroy.txt", 
                        help="Chemin du fichier d'entrée nettoyé")
    parser.add_argument("--output", default="data/processed/antandroy_dataset.jsonl", 
                        help="Chemin du fichier de sortie JSONL")
    parser.add_argument("--window", type=int, default=1, 
                        help="Nombre de lignes à utiliser comme contexte")
    parser.add_argument("--test-split", type=float, default=0.1, 
                        help="Proportion des données à réserver pour le test (0-1)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Graine pour la reproductibilité")
    args = parser.parse_args()
    
    # Création du dataset
    result = create_dataset(
        args.input, 
        args.output,
        window_size=args.window,
        test_split=args.test_split,
        seed=args.seed
    )