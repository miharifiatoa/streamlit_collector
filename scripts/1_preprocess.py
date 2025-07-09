import os
import re
import argparse

# Pré-compilation des expressions régulières
punct_regex = re.compile(r'[^\w\sʼàâèéêîôùû]')
space_regex = re.compile(r'\s+')

def clean_text(text):
    """Nettoie une chaîne de texte en malgache."""
    text = text.lower()
    text = punct_regex.sub('', text)  # caractères valides + diacritiques malgaches
    text = space_regex.sub(' ', text).strip()
    return text

def preprocess_corpus(input_path, output_path):
    """Prétraite un corpus ligne par ligne avec statistiques."""
    total_lines = 0
    empty_lines = 0
    total_chars_before = 0
    total_chars_after = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            total_lines += 1
            if not line.strip():
                empty_lines += 1
                continue
                
            total_chars_before += len(line)
            clean_line = clean_text(line)
            total_chars_after += len(clean_line)
            outfile.write(clean_line + '\n')
    
    print(f"✔ Fichier nettoyé enregistré dans : {output_path}")
    print(f"  - Lignes traitées: {total_lines}")
    print(f"  - Lignes vides ignorées: {empty_lines}")
    print(f"  - Lignes conservées: {total_lines - empty_lines}")
    if total_chars_before > 0:
        print(f"  - Réduction taille: {total_chars_before} → {total_chars_after} caractères ({(1 - total_chars_after/total_chars_before)*100:.1f}%)")
    else:
        print("  - Fichier d'entrée vide")

if __name__ == "__main__":
    # Configuration du parser d'arguments
    parser = argparse.ArgumentParser(description="Nettoie un corpus de texte malgache")
    parser.add_argument("--input", default="data/raw/texte.txt", 
                        help="Chemin du fichier d'entrée")
    parser.add_argument("--output", default="data/clean/clean_texte_antandroy.txt", 
                        help="Chemin du fichier de sortie")
    args = parser.parse_args()
    
    # Création du répertoire de sortie si nécessaire
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Traitement du corpus
    preprocess_corpus(args.input, args.output)