# Chatbot Antandroy – Deep Learning & NLP

Ce projet vise à développer un chatbot et des outils de traitement automatique pour la langue Antandroy (dialecte du sud de Madagascar), en utilisant des techniques modernes de deep learning et de NLP.

---

## 1. Présentation

- **Langue cible** : Antandroy (famille malgache)
- **Fonctionnalités** :
  - Chatbot multilingue (Antandroy/Français)
  - Éditeur de dataset (ajout, édition, collecte de voix)
  - Collecte et prétraitement de textes
  - Interface web interactive (Streamlit)
- **Objectif** : Créer des ressources et modèles pour une langue à faibles ressources

---
.env┌──(.env)─(miharifiatoa㉿slem)-[~/code_source/Projet M2 GID 2024-2024 /ML-DL]
└─$ git add .    
^C
                                                                                                                             
.env┌──(.env)─(miharifiatoa㉿slem)-[~/code_source/Projet M2 GID 2024-2024 /ML-DL]
└─$ git add .
^C
                                                                                                                             
.env┌──(.env)─(miharifiatoa㉿slem)-[~/code_source/Projet M2 GID 2024-2024 /ML-DL]
└─$ git add  
Rien de spécifié, rien n'a été ajouté.
astuce : Vous vouliez sûrement utiliser 'git add .' ?
astuce : Disable this message with "git config advice.addEmptyPathspec false"
                                                                                                                             
.env┌──(.env)─(miharifiatoa㉿slem)-[~/code_source/Projet M2 GID 2024-2024 /ML-DL]
└─$ 
## 2. Structure du projet

```
ML-DL/
├── data/
│   ├── raw/                # Textes bruts collectés
│   ├── clean/              # Textes nettoyés
│   ├── processed/          # Dataset final pour entraînement
│   └── voices/             # Enregistrements vocaux collectés
├── models/                 # Modèles fine-tunés (ex: mt5_antandroy)
├── scripts/
│   ├── 6_streamlit_chatbot.py   # Application principale Streamlit
│   └── requirements.txt
├── pretraitement.py        # Script de prétraitement (auto-généré si absent)
├── Dockerfile
└── README.md
```

---

## 3. Installation locale

### Prérequis

- Python 3.8+
- `pip` (gestionnaire de paquets Python)
- (Optionnel) `ffmpeg` pour l'audio

### Installation des dépendances

```bash
cd scripts
pip install -r requirements.txt
```

### Lancement de l'application

```bash
streamlit run scripts/6_streamlit_chatbot.py
```

L'application sera accessible sur [http://localhost:8501](http://localhost:8501).

---

## 4. Utilisation avec Docker

### Construction de l'image

```bash
docker build -t antandroy-streamlit-app .
```

### Lancement du conteneur

```bash
docker run -p 8501:8501 -v $(pwd)/data:/app/data antandroy-streamlit-app
```

- Accès via [http://localhost:8501](http://localhost:8501)
- Les données (datasets, voix) sont persistées dans le dossier `data/` du projet

---

## 5. Configuration

- **Modèle** : Placez votre modèle fine-tuné dans `models/mt5_antandroy`
- **Dataset** : Le fichier principal est `data/processed/output.json`
- **Prétraitement** : Le script `pretraitement.py` est généré automatiquement si absent

---

## 6. Dépendances principales

Voir `scripts/requirements.txt` :
```
streamlit
transformers
torch
audio-recorder-streamlit
nest_asyncio
sentencepiece
datasets
pandas
numpy
tqdm
```

---

## 7. Fonctionnalités principales

- **Chatbot** : Posez des questions en Antandroy ou en français, obtenez des réponses générées par un modèle MT5 spécialisé.
- **Éditeur de dataset** : Parcourez, éditez, enrichissez le dictionnaire Antandroy-Français, ajoutez des exemples et enregistrez votre voix.
- **Collecte de textes** : Ajoutez de nouveaux textes, traitez-les et enrichissez le dataset.

---

## 8. Conseils pour le prétraitement et le fine-tuning

- Voir la section détaillée dans ce README pour les étapes de prétraitement, la création de tokenizer, et le fine-tuning sur des modèles multilingues (ex : mT5, XLM-RoBERTa, BLOOM).
- Travaillez avec des locuteurs natifs pour valider la qualité linguistique.

---

## 9. Ressources complémentaires

- Bibliothèques : HuggingFace Transformers, SentencePiece, PyTorch, Streamlit
- Outils recommandés : ffmpeg (audio), pandas, numpy
- Documentation linguistique : dictionnaires malgaches, grammaires régionales

---

## 10. Contact

Pour toute question ou contribution, contactez l’équipe projet ou ouvrez une issue sur le dépôt GitHub.

---

**Un apprentissage progressif et itératif sera crucial pour développer un modèle performant en langue antandroy.**

Cours d'Entraînement en Langue Antandroy pour Deep Learning
1. Introduction à la Langue Antandroy
1.1 Contexte Linguistique
    • Langue parlée par les Antandroy, un groupe ethnique du sud de Madagascar 
    • Appartient à la famille des langues malgaches 
    • Dialecte régional avec des caractéristiques linguistiques uniques 
1.2 Caractéristiques Linguistiques
    • Structure morphologique riche 
    • Forte présence de préfixes et suffixes 
    • Ordre des mots flexible 
    • Nombreuses variations contextuelles 
2. Analyse Linguistique du Corpus
2.1 Analyse Lexicale
    • Vocabulaire diversifié observé dans le jeu de données 
    • Mots courts et longs 
    • Présence de motifs récurrents linguistiques 
Exemples de Structures Lexicales
    • Préfixes fréquents : 
        ◦ an-, aha-, aka-, am- 
    • Suffixes communs : 
        ◦ -tse, -tsy, -na 
    • Mots composés : 
        ◦ anak' ondry (exemple de construction) 
        ◦ avy avao (expression de mouvement) 
2.2 Morphologie
    • Analyse des transformations morphologiques 
    • Identification des règles de dérivation et flexion 
    • Étude des variations grammaticales 
3. Préparation des Données pour Deep Learning
3.1 Prétraitement du Corpus
    • Nettoyage des données 
    • Normalisation 
    • Tokenization 
    • Gestion des variations orthographiques 
Techniques de Prétraitement
    • Segmentation en tokens 
    • Création de vocabulaire 
    • Encodage des caractéristiques linguistiques 
    • Gestion des mots rares/uniques 
3.2 Architectures de Modèles Recommandées
    • Réseaux de neurones récurrents (RNN) 
    • Long Short-Term Memory (LSTM) 
    • Transformers adaptés aux langues à morphologie complexe 
    • Modèles hybrides caractère-mot 
4. Stratégies d'Entraînement
4.1 Objectifs d'Apprentissage
    • Génération de texte 
    • Compréhension contextuelle 
    • Traduction 
    • Reconnaissance de motifs linguistiques 
4.2 Approches d'Entraînement
    • Apprentissage supervisé 
    • Apprentissage non supervisé 
    • Transfer learning 
    • Fine-tuning sur corpus spécifique 
5. Défis et Considérations Spécifiques
5.1 Particularités Linguistiques
    • Gestion des variations dialectales 
    • Compréhension des nuances contextuelles 
    • Traitement des constructions grammaticales complexes 
5.2 Limitations Potentielles
    • Corpus limité 
    • Manque de ressources linguistiques standardisées 
    • Variations régionales importantes 
6. Métriques d'Évaluation
6.1 Métriques Quantitatives
    • Perplexité 
    • BLEU score 
    • Précision lexicale 
    • Cohérence syntaxique 
6.2 Métriques Qualitatives
    • Évaluation par des locuteurs natifs 
    • Test de compréhension contextuelle 
    • Analyse de la génération de texte 
7. Ressources Complémentaires
7.1 Outils Recommandés
    • Bibliothèques de NLP : 
        ◦ Transformers 
        ◦ Tensorflow 
        ◦ PyTorch 
    • Outils de tokenization adaptés 
7.2 Références Linguistiques
    • Dictionnaires malgaches 
    • Grammaires régionales 
    • Documents ethnolinguistiques 
Un apprentissage progressif et itératif sera crucial pour développer un modèle performant en langue antandroy.

Stratégie d'Apprentissage Progressif et Itératif pour Modèle de Langue Antandroy
1. Phases d'Apprentissage Progressif
1.1 Phase Initiale : Compréhension Fondamentale
    • Objectif : Établir une base linguistique solide 
    • Étapes clés : 
        1. Analyse lexicale approfondie 
        2. Identification des structures grammaticales de base 
        3. Création d'un dictionnaire de tokens 
Techniques de Construction Initiale
    • Segmentation des mots en tokens 
    • Analyse des fréquences lexicales 
    • Cartographie des structures morphologiques 
1.2 Phase Intermédiaire : Apprentissage Contextuel
    • Objectif : Développer la compréhension contextuelle 
    • Techniques : 
        1. Modèles de prédiction de tokens 
        2. Apprentissage des variations contextuelles 
        3. Reconnaissance des motifs linguistiques 
Mécanismes d'Enrichissement
    • Intégration de contextes courts 
    • Apprentissage des liens syntaxiques 
    • Modélisation des transitions grammaticales 
1.3 Phase Avancée : Génération et Raffinement
    • Objectif : Générer du contenu linguistiquement cohérent 
    • Stratégies : 
        1. Génération de phrases 
        2. Validation par des locuteurs natifs 
        3. Ajustement fin (fine-tuning) 
2. Approche Itérative de Développement
2.1 Cycle d'Amélioration Continue
Copier
[Collecte de Données] → [Prétraitement] → [Entraînement] → [Évaluation] → [Raffinement]
                          ↑_______________________|
2.2 Mécanismes Itératifs
    • Boucle de Rétroaction : 
        1. Évaluation des performances 
        2. Identification des lacunes 
        3. Ajustement des paramètres 
        4. Réentraînement ciblé 
2.3 Techniques de Validation
    • Tests de compréhension graduels 
    • Évaluation par experts linguistiques 
    • Comparaison avec corpus de référence 
3. Techniques d'Apprentissage Spécialisées
3.1 Transfer Learning Adaptatif
    • Utilisation de modèles pré-entraînés sur langues apparentées 
    • Adaptation progressive aux spécificités antandroy 
3.2 Apprentissage Semi-Supervisé
    • Combinaison de données étiquetées et non étiquetées 
    • Maximisation de l'utilisation des ressources limitées 
4. Gestion des Défis Spécifiques
4.1 Gestion de la Rareté des Données
    • Stratégies : 
        1. Augmentation synthétique de données 
        2. Techniques de régularisation 
        3. Modèles de génération de données 
4.2 Adaptation Contextuelle
    • Intégration de connaissances culturelles 
    • Compréhension des nuances dialectales 
5. Métriques de Progression
5.1 Indicateurs de Performance
    • Réduction de la perplexité 
    • Augmentation de la cohérence grammaticale 
    • Précision de la génération contextuelle 
5.2 Évaluation Qualitative
    • Tests de compréhension par locuteurs natifs 
    • Analyses linguistiques approfondies 
Conclusion : Flexibilité et Adaptation
L'apprentissage d'un modèle de langue antandroy n'est pas un processus linéaire mais un voyage d'exploration continue, nécessitant :
    • Patience 
    • Adaptabilité 
    • Expertise linguistique 
    • Approche technologique rigoureuse

Pour prétraiter un texte écrit en antandroy (un dialecte malgache parlé dans le sud de Madagascar) en vue d'entraîner un modèle LLM, voici les étapes que recommandé:
    1. Normalisation des caractères spéciaux - Standardiser les caractères spécifiques à l'antandroy et les diacritiques 
    2. Tokenisation adaptée - Créer des tokens qui respectent les particularités morphologiques de l'antandroy 
    3. Nettoyage des données - Éliminer les erreurs de saisie, les incohérences orthographiques et standardiser la transcription 
    4. Segmentation - Diviser le texte en phrases et en mots selon les règles grammaticales de l'antandroy 
    5. Analyse morphologique - Identifier les racines des mots et les affixes, très importants dans les langues malgaches 
    6. Gestion des emprunts linguistiques - Traiter correctement les mots empruntés au français ou à d'autres langues 
    7. Augmentation de données - Générer des variations syntaxiques valides pour enrichir le corpus d'entraînement 
    8. Création d'un dictionnaire spécifique - Développer un lexique de référence pour l'antandroy 
Il est crucial de travailler avec des locuteurs natifs pour valider la qualité linguistique du prétraitement. Comme l'antandroy est une langue à faibles ressources, vous pourriez également envisager un transfert d'apprentissage depuis des modèles entraînés sur d'autres dialectes malgaches ou sur le malgache officiel.
Souhaitez-vous que je développe certains aspects spécifiques du prétraitement ?
M
Oui, développe certains aspects spécifiques du prétraitement, en faite j'ai deja des textes, des redactions, des commentaires ecrit en antandroy, il faut m'apprendre ce que je dois faire pour le pretraitement et la transfert learning, en me proposant une model preetrainer gratuit...
2/2
Je vais développer les aspects spécifiques du prétraitement pour vos textes en antandroy et vous proposer une approche de transfert d'apprentissage avec des modèles pré-entraînés gratuits.
Prétraitement détaillé pour textes en antandroy
1. Normalisation orthographique
    • Créez un dictionnaire de correspondance pour les variations orthographiques courantes 
    • Standardisez les formes écrites des mots qui peuvent avoir plusieurs graphies 
    • Exemple : si un même mot apparaît comme "ampela" et "ampella", standardisez vers une seule forme 
2. Tokenisation spécifique
    • L'antandroy, comme les autres dialectes malgaches, utilise des affixes complexes 
    • Créez un tokeniseur qui respecte les structures morphologiques : 
        ◦ Préfixes (mi-, maha-, mana-, etc.) 
        ◦ Infixes 
        ◦ Suffixes (-ana, -ina, etc.) 
    • Utilisez SentencePiece ou WordPiece pour générer des sous-mots adaptés à la structure de l'antandroy 
3. Nettoyage contextuel
    • Supprimez les codes de formatage, métadonnées et autres éléments non linguistiques 
    • Normalisez la ponctuation et les chiffres 
    • Traitez les emprunts du français ou d'autres langues (décidez si vous les conservez ou les adaptez) 
4. Préparation des données d'entraînement
    • Divisez votre corpus en ensembles d'entraînement/validation/test (70%/15%/15%) 
    • Pour les rédactions et commentaires, préservez l'intégrité des documents complets 
    • Créez des paires de questions-réponses si vous avez des textes conversationnels 
Transfert d'apprentissage avec modèles pré-entraînés gratuits
Modèles pré-entraînés recommandés (gratuits)
    1. AfroLM : Spécialisé pour les langues africaines, pourrait offrir des bases utiles 
    2. MasakhaNER : Bien qu'axé sur la reconnaissance d'entités, contient des embeddings pour certaines langues africaines 
    3. XLM-RoBERTa : Modèle multilingue de Facebook qui pourrait avoir une certaine couverture pour les langues malgaches 
    4. BLOOM : Modèle multilingue open-source avec capacités pour plusieurs langues africaines 
Procédure de transfert d'apprentissage
    1. Choisissez le modèle de base (comme XLM-RoBERTa ou BLOOM) 
    2. Adaptez la couche d'embedding pour inclure les tokens spécifiques à l'antandroy 
    3. Utilisez une technique d'adaptation progressive : 
        ◦ D'abord, entraînez sur du malgache standard (si disponible) 
        ◦ Puis, affinez sur votre corpus antandroy 
Implémentation pratique avec Hugging Face
python
Copier
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch

# Chargez un modèle pré-entraîné (par exemple XLM-RoBERTa)
model_name = "xlm-roberta-base"  # ou "bloom-560m" pour un modèle plus petit
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Adaptez le tokenizer à l'antandroy (ajoutez des tokens spécifiques)
# Basé sur votre analyse morphologique préalable
special_tokens = ["<tokens_spécifiques_antandroy>"]  # À remplacer avec vos tokens réels
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
model.resize_token_embeddings(len(tokenizer))

# Configuration de l'entraînement
training_args = TrainingArguments(
    output_dir="./antandroy-model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

# Entraînement du modèle
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15),
    train_dataset=your_preprocessed_dataset,  # Remplacer par votre dataset prétraité
)

trainer.train()
Évaluation et ajustement
    • Utilisez des métriques comme la perplexité sur votre ensemble de validation 
    • Ajustez les hyperparamètres en fonction des résultats 
    • Validez avec des locuteurs natifs pour vérifier la qualité linguistique 
Ces étapes vous permettront de créer un modèle spécifique à l'antandroy en tirant parti des modèles pré-entraînés gratuits disponibles. Le plus grand défi sera probablement l'adaptation du tokeniseur pour qu'il représente correctement les caractéristiques morphologiques de l'antandroy.


Étape	Description
1. Collecte des données	Rassembler un corpus Antandroy (textes, dialogues, proverbes, etc.)
2. Nettoyage et prétraitement	Tokenisation, nettoyage, normalisation du texte
3. Construction du dataset	Formatage en paires question-réponse ou séquences
4. Tokenisation et vocabulaire	Créer un vocabulaire, tokenizer personnalisé
5. Choix du modèle	GPT-like (transformer), Seq2Seq, ou fine-tuning d’un LLM léger
6. Entraînement	Modèle entraîné sur ton corpus, avec attention à l’overfitting
7. Évaluation	Qualité des réponses : BLEU score, test humain, perplexité
8. Déploiement	Via une API Flask ou une app web
9. Interface de test	Web app ou ligne de commande pour tester le bot


Sprint 1 : Collecte & Prétraitement des données
Durée estimée : 1 semaine
Objectif : Obtenir un corpus propre Antandroy
User Stories / Cartes :
    • 📥 Collecter les textes Antandroy (pages, journaux, fichiers locaux, voix à transcrire…)
    • 🧼 Nettoyer le texte (ponctuation, mots étrangers, diacritiques si nécessaires)
    • ✂️ Découper les textes en lignes utilisables
    • 📊 Analyse simple : longueur moyenne des phrases, vocabulaire, fréquence
    • 💬 (optionnel) Traduction ligne par ligne pour corpus parallèle

🟩 Sprint 2 : Dataset & Tokenizer
Durée estimée : 1 semaine
Objectif : Créer un dataset prêt pour l'entraînement
User Stories / Cartes :
    • 📁 Formater les données : paires (input → output) ou texte continu
    • 🔤 Créer un vocabulaire adapté à la langue Antandroy
    • 🔧 Implémenter un tokenizer BPE ou WordPiece si personnalisé
    • 📉 Limiter / filtrer les phrases trop longues / trop courtes

🟨 Sprint 3 : Modèle & Entraînement
Durée estimée : 2 semaines
Objectif : Entraîner le modèle à répondre en Antandroy
User Stories / Cartes :
    • 🧠 Choisir une architecture simple (ex: GPT-2 Small, RNN, LSTM Seq2Seq)
    • 🛠️ Adapter le modèle au vocabulaire Antandroy
    • 🚀 Lancer l’entraînement, suivi des logs (loss, perplexité)
    • 📈 Évaluer avec validation, quelques phrases de test manuelles
    • 💾 Sauvegarder le modèle entraîné

🟧 Sprint 4 : Déploiement et Test utilisateur
Durée estimée : 1 semaine
Objectif : Tester le chatbot avec des vrais utilisateurs
User Stories / Cartes :
    • 🌍 Créer une interface (terminal ou web app Flask / React)
    • 🧪 Laisser des utilisateurs tester et donner leur feedback
    • 🗣️ Itérer sur les erreurs linguistiques du modèle
    • 🎓 Ajout d’un mini module pour reconnaître le dialecte (facultatif)
    • 📚 Documentation et publication du modèle

📚 Tech Stack proposé
Domaine	Outils
Prétraitement	Python, NLTK / spaCy, regex
Modèle	PyTorch / HuggingFace Transformers
Tokenisation	SentencePiece / Tokenizer HuggingFace
Déploiement	Flask / FastAPI / Streamlit
Suivi	TensorBoard, Weights & Biases
Gestion agile	Trello, Jira, Notion

📌 Conseils spécifiques pour une langue rare comme Antandroy
    • Corpus faible ? : Utilise du data augmentation (reformulation, permutation, mix traduction)
    • Pas de tokenizer prêt ? : Entraîne un tokenizer SentencePiece à partir de ton corpus
    • Pas assez de données ? : Fais du fine-tuning sur un modèle multilingue préentraîné (ex : mT5, mGPT)


🟦 Sprint 1 – Collecte & Prétraitement des données (1 semaine)
🎯 Objectif : Obtenir un corpus propre en Antandroy
📥 Collecte
    • Récupérer des textes Antandroy (fichiers, PDF, blogs, documents oraux…)
    • Identifier des locuteurs pour des transcriptions manuelles (facultatif)
    • Télécharger ou transcrire des extraits de chansons, contes, proverbes, etc.
🧼 Nettoyage
    • Enlever les caractères inutiles (punctuation, HTML, emojis)
    • Supprimer les mots étrangers ou les marquer (<fr>, <mg>, etc.)
    • Normaliser les textes (minuscule, accents)
✂️ Segmenter
    • Découper les textes en phrases / lignes
    • Enregistrer dans un fichier .txt ou .csv avec une ligne par phrase
📊 Analyse exploratoire
    • Calculer la longueur moyenne des phrases
    • Générer une fréquence des mots
    • Identifier les mots les plus rares / les plus fréquents
💬 (Optionnel) Traduction
    • Traduire quelques phrases en français ou malgache standard pour alignement
    • Stocker dans une structure {"source": "...", "cible": "..."}

🟩 Sprint 2 – Dataset & Tokenizer (1 semaine)
🎯 Objectif : Créer un dataset utilisable pour l’entraînement
📁 Dataset
    • Organiser les données : paires Q-R ou texte continu
    • Créer un fichier JSONL, CSV ou texte brut
🔤 Vocabulaire
    • Générer le vocabulaire (avec seuil de fréquence min)
    • Gérer les tokens spéciaux (<pad>, <unk>, <eos>…)
🔧 Tokenizer
    • Entraîner un tokenizer SentencePiece sur ton corpus
    • Vérifier la segmentation sur des phrases de test
    • Sauvegarder et documenter le tokenizer
📉 Filtrage
    • Supprimer les phrases très longues ou trop courtes
    • Garder un équilibre entre complexité et cohérence

🟨 Sprint 3 – Modèle & Entraînement (2 semaines)
🎯 Objectif : Créer un modèle qui comprend et génère de l’Antandroy
🧠 Architecture
    • Choisir un modèle : GPT-2, RNN avec attention, Transformer léger
    • Adapter à une architecture de génération de texte
🛠️ Préparation
    • Charger le tokenizer
    • Adapter le modèle pour un vocabulaire personnalisé
    • Préparer les données en format input_ids, labels
🚀 Entraînement
    • Lancer les sessions avec suivi (logs, loss, epoch)
    • Sauvegarder régulièrement le modèle (checkpoints)
📈 Évaluation
    • Créer un script pour tester manuellement
    • Calculer la perplexité, BLEU ou score qualitatif
💾 Sauvegarde
    • Sauvegarder le modèle final
    • Exporter vers Hugging Face ou un répertoire local

🟧 Sprint 4 – Déploiement & Test utilisateur (1 semaine)
🎯 Objectif : Utilisation réelle du chatbot
🌍 Interface
    • Créer une petite app Flask ou Streamlit
    • Ajouter un champ "Pose ta question en Antandroy"
🧪 Tests utilisateur
    • Collecter des retours de locuteurs natifs
    • Enregistrer les réponses correctes vs incorrectes
🗣️ Amélioration
    • Ajuster le modèle ou les données selon les feedbacks
    • Ajouter un système de correction interactive
🎓 Détection du dialecte (optionnel)
    • Ajouter une étape de classification Antandroy vs Sakalava vs Bara, etc.
📚 Documentation
    • Rédiger une documentation du projet (README, usage)
    • Publier sur GitHub ou Hugging Face si souhaité


Structure du projet :
chatbot-antandroy/
│
├── data/
│   ├── raw/                # Textes bruts collectés
│   ├── clean/              # Textes nettoyés
│   ├── processed/          # Dataset final pour entraînement
│   └── tokenizer/          # Tokenizer SentencePiece
│
├── scripts/
│   ├── 1_preprocess.py
│   ├── 2_train_tokenizer.py
│   └── 3_prepare_dataset.py
│
├── README.md
└── requirements.txt

