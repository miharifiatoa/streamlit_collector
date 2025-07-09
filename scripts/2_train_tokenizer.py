import sentencepiece as spm
import os
import argparse
import time

def train_tokenizer(input_file, model_prefix, vocab_size=8000, model_type='bpe', 
                    character_coverage=0.9995, input_sentence_size=1000000):
    """
    Entraîne un tokenizer SentencePiece avec des paramètres configurables.
    
    Args:
        input_file: Chemin vers le fichier de corpus nettoyé
        model_prefix: Préfixe pour le modèle de tokenizer (sans extension)
        vocab_size: Taille du vocabulaire à générer
        model_type: Type de modèle ('bpe' ou 'unigram')
        character_coverage: Couverture des caractères (utile pour les langues avec beaucoup de caractères)
        input_sentence_size: Nombre maximum de phrases à utiliser pour l'entraînement
    """
    start_time = time.time()
    
    # Vérification de l'existence du fichier d'entrée
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Fichier d'entrée introuvable: {input_file}")
    
    # Calcul de la taille du fichier pour statistiques
    file_size = os.path.getsize(input_file) / (1024 * 1024)  # en MB
    
    print(f"⏳ Démarrage de l'entraînement du tokenizer...")
    print(f"  - Fichier d'entrée: {input_file} ({file_size:.2f} MB)")
    print(f"  - Type de modèle: {model_type}")
    print(f"  - Taille de vocabulaire: {vocab_size}")
    
    # Configuration de l'entraînement
    train_args = {
        'input': input_file,
        'model_prefix': model_prefix,
        'vocab_size': vocab_size,
        'model_type': model_type,
        'pad_id': 0,
        'unk_id': 1,
        'bos_id': 2,
        'eos_id': 3,
        'user_defined_symbols': ["<sep>", "<pad>"],
        'character_coverage': character_coverage,
        'input_sentence_size': input_sentence_size,
        'shuffle_input_sentence': True,
        'normalization_rule_name': 'identity'  # Pas de normalisation supplémentaire
    }
    
    # Entraînement du modèle
    spm.SentencePieceTrainer.train(**train_args)
    
    # Validation du modèle généré
    if os.path.exists(f"{model_prefix}.model") and os.path.exists(f"{model_prefix}.vocab"):
        model_size = os.path.getsize(f"{model_prefix}.model") / 1024  # en KB
        vocab_size = os.path.getsize(f"{model_prefix}.vocab") / 1024  # en KB
        
        # Test rapide du tokenizer
        sp = spm.SentencePieceProcessor()
        sp.load(f"{model_prefix}.model")
        
        training_time = time.time() - start_time
        
        print(f"✔ Tokenizer entraîné avec succès en {training_time:.2f} secondes")
        print(f"  - Modèle: {model_prefix}.model ({model_size:.2f} KB)")
        print(f"  - Vocabulaire: {model_prefix}.vocab ({vocab_size:.2f} KB)")
        print(f"  - Taille réelle du vocabulaire: {sp.get_piece_size()} tokens")
        
        # Exemple d'utilisation (si le fichier d'entrée contient du texte)
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                sample_text = f.readline().strip()
                if sample_text:
                    tokens = sp.encode_as_pieces(sample_text)
                    print(f"\nExemple de tokenisation:")
                    print(f"  - Texte: '{sample_text}'")
                    print(f"  - Tokens: {tokens}")
        except Exception as e:
            print(f"Note: Impossible de charger un exemple de texte: {e}")
    else:
        print(f"⚠️ Problème lors de la création du tokenizer")

if __name__ == "__main__":
    # Configuration du parser d'arguments
    parser = argparse.ArgumentParser(description="Entraîne un tokenizer SentencePiece pour le malgache")
    parser.add_argument("--input", default="data/clean/clean_texte_antandroy.txt", 
                        help="Chemin du fichier d'entrée nettoyé")
    parser.add_argument("--output", default="data/tokenizer/antandroy_tokenizer", 
                        help="Préfixe pour les fichiers du tokenizer (sans extension)")
    parser.add_argument("--vocab-size", type=int, default=8000, 
                        help="Taille du vocabulaire à générer")
    parser.add_argument("--model-type", choices=['bpe', 'unigram'], default='bpe',
                        help="Type de modèle de tokenisation (bpe ou unigram)")
    parser.add_argument("--character-coverage", type=float, default=0.9995,
                        help="Couverture des caractères (entre 0 et 1)")
    args = parser.parse_args()
    
    # Création du répertoire de sortie si nécessaire
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Entraînement du tokenizer
    train_tokenizer(
        args.input, 
        args.output, 
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage
    )