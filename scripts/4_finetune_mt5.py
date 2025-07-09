import os
import argparse
import logging
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import traceback

import torch
from datasets import Dataset, DatasetDict, load_dataset
from evaluate import load
from transformers import (
    MT5ForConditionalGeneration,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    set_seed,
    __version__ as transformers_version
)
from transformers.trainer_utils import get_last_checkpoint
import packaging.version as pv
from tqdm import tqdm

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def safe_execute(func):
    """Décorateur pour une exécution sécurisée des fonctions."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Erreur dans {func.__name__}: {str(e)}\n{error_trace}")
            raise
    return wrapper

class CustomMetrics:
    """Classe pour calculer les métriques d'évaluation."""
    def __init__(self):
        try:
            self.bleu = load("sacrebleu")
            self.rouge = load("rouge")
            self.meteor = load("meteor")
            logger.info("Métriques chargées avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des métriques: {e}")
            raise
        
    def compute_metrics(self, pred_str: List[str], label_str: List[str]) -> Dict[str, float]:
        """Calcule plusieurs métriques pour évaluer la qualité de la traduction."""
        if not pred_str or not label_str:
            logger.error("Aucune prédiction ou référence pour calculer les métriques")
            return {"error": "Données insuffisantes"}
            
        try:
            # Structure pour les métriques BLEU (prend une liste de références)
            references_list = [[ref] for ref in label_str]
            bleu_score = self.bleu.compute(predictions=pred_str, references=references_list)["score"]
            
            # ROUGE prend des listes simples
            rouge_scores = self.rouge.compute(
                predictions=pred_str,
                references=label_str,
                use_stemmer=True
            )
            
            # METEOR calcule la moyenne des scores pour chaque paire
            meteor_scores = []
            for pred, ref in zip(pred_str, label_str):
                try:
                    meteor_score = self.meteor.compute(predictions=[pred], references=[ref])["meteor"]
                    meteor_scores.append(meteor_score)
                except:
                    # Si METEOR échoue pour une paire, on continue avec les autres
                    continue
                    
            # Calculer la moyenne des scores METEOR
            meteor_avg = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
            
            return {
                "bleu": bleu_score,
                "rouge1": rouge_scores["rouge1"],
                "rouge2": rouge_scores["rouge2"],
                "rougeL": rouge_scores["rougeL"],
                "meteor": meteor_avg
            }
        except Exception as e:
            logger.error(f"Erreur lors du calcul des métriques: {e}")
            return {"error": str(e)}

class TextGenerator:
    """Classe pour générer des traductions."""
    def __init__(self, model, tokenizer, device, generation_params=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        # Paramètres de génération par défaut avec fallback
        self.generation_params = generation_params or {
            "max_length": 128,
            "num_beams": 4,
            "early_stopping": True,
            "no_repeat_ngram_size": 2,
            "length_penalty": 1.0,
            "temperature": 0.7
        }
        logger.info(f"Générateur de texte initialisé sur {device}")
        
    @safe_execute
    def generate(self, input_text: str, max_length: int = None) -> str:
        """Génère une traduction pour un texte d'entrée."""
        if not input_text or not input_text.strip():
            logger.warning("Tentative de génération avec un texte vide")
            return ""
            
        # Utiliser max_length spécifié ou celui par défaut
        max_length = max_length or self.generation_params["max_length"]
        
        # Tokenisation de l'entrée
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )
        
        # Déplacer les entrées vers le dispositif cible
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            
        # Paramètres de génération dynamiques
        gen_params = self.generation_params.copy()
        gen_params["max_length"] = max_length
        
        # Génération de la sortie
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_params
            )
            
        # Décodage et nettoyage de la sortie
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    def update_generation_params(self, new_params: Dict[str, Any]):
        """Met à jour les paramètres de génération."""
        self.generation_params.update(new_params)
        logger.info(f"Paramètres de génération mis à jour: {new_params}")

class ModelValidator:
    """Classe pour valider le modèle."""
    def __init__(self, model, tokenizer, device, metrics):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.metrics = metrics
        self.generator = TextGenerator(model, tokenizer, device)
    
    @safe_execute
    def validate(self, validation_data_path, num_examples: int = 5) -> Dict[str, Any]:
        """Valide le modèle sur un ensemble de données.
        
        Args:
            validation_data_path: Chemin vers les données de validation (fichier ou liste)
            num_examples: Nombre d'exemples à valider
            
        Returns:
            Un dictionnaire contenant les exemples et les métriques
        """
        self.model.eval()
        results = []
        metrics_data = {"predictions": [], "references": []}
        
        # Charger les données de validation selon le type d'entrée
        validation_data = []
        if isinstance(validation_data_path, str):
            logger.info(f"Chargement des données de validation depuis {validation_data_path}")
            try:
                # Vérifier si le fichier existe
                if not os.path.exists(validation_data_path):
                    logger.error(f"Fichier de validation introuvable: {validation_data_path}")
                    return {"examples": [], "metrics": {}, "error": f"Fichier introuvable: {validation_data_path}"}
                
                # Déterminer le format du fichier
                if validation_data_path.endswith('.jsonl'):
                    # Format JSONL - une ligne JSON par exemple
                    with open(validation_data_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue  # Ignorer les lignes vides
                                
                            try:
                                example = json.loads(line)
                                validation_data.append(example)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Ligne {line_num} ignorée - JSON invalide: {line[:50]}... Erreur: {e}")
                elif validation_data_path.endswith('.json'):
                    # Format JSON - un tableau ou objet JSON
                    with open(validation_data_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            if isinstance(data, list):
                                validation_data = data
                            elif isinstance(data, dict) and 'examples' in data:
                                validation_data = data['examples']
                            else:
                                logger.warning("Format JSON inattendu, tentative de traitement comme un seul exemple")
                                validation_data = [data]
                        except json.JSONDecodeError as e:
                            logger.error(f"Erreur lors du chargement du fichier JSON: {e}")
                            return {"examples": [], "metrics": {}, "error": f"JSON invalide: {e}"}
                else:
                    # Tenter de traiter comme JSONL par défaut
                    with open(validation_data_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    validation_data.append(json.loads(line))
                                except json.JSONDecodeError:
                                    # Si pas JSON, essayer de traiter comme texte brut avec séparateur
                                    parts = line.split('\t')
                                    if len(parts) >= 2:
                                        validation_data.append({"input": parts[0], "output": parts[1]})
                                    else:
                                        logger.warning(f"Ligne ignorée - format invalide: {line[:50]}...")
            except Exception as e:
                logger.error(f"Erreur lors de la lecture du fichier de validation: {e}")
                return {"examples": [], "metrics": {}, "error": str(e)}
        elif isinstance(validation_data_path, (list, Dataset)):
            # Si c'est déjà une liste d'exemples ou un dataset Hugging Face
            validation_data = validation_data_path
        else:
            logger.error(f"Type de données de validation non pris en charge: {type(validation_data_path)}")
            return {"examples": [], "metrics": {}, "error": "Format de données non supporté"}
        
        # Convertir Dataset en liste si nécessaire
        if hasattr(validation_data, '__len__') and hasattr(validation_data, '__getitem__'):
            if not isinstance(validation_data, list):
                try:
                    validation_data = list(validation_data)
                except Exception as e:
                    logger.error(f"Erreur lors de la conversion des données: {e}")
                    return {"examples": [], "metrics": {}, "error": str(e)}
        
        logger.info(f"Données de validation chargées: {len(validation_data)} exemples")
        
        # S'assurer qu'il y a des données à valider
        if not validation_data:
            logger.error("Aucune donnée de validation n'a été chargée correctement.")
            return {"examples": [], "metrics": {}, "error": "Aucune donnée valide"}
        
        # Limiter le nombre d'exemples à traiter
        examples_to_process = validation_data[:num_examples] if num_examples > 0 else validation_data
        logger.info(f"Validation sur {len(examples_to_process)} exemples")
        
        valid_examples = 0
        for i, example in enumerate(tqdm(examples_to_process, desc="Validation")):
            try:
                # Vérifier si l'exemple est valide et complet
                if not example:
                    logger.warning(f"Exemple {i} vide, ignoré.")
                    continue
                    
                # Assurer que les clés requises sont présentes
                if not isinstance(example, dict):
                    logger.warning(f"Exemple {i} n'est pas un dictionnaire: {type(example)}")
                    continue
                
                # Vérifier les différentes nomenclatures possibles des champs
                input_field = next((f for f in ['input', 'source', 'french', 'fr'] if f in example), None)
                output_field = next((f for f in ['output', 'target', 'malagasy', 'mg'] if f in example), None)
                
                if not input_field or not output_field:
                    logger.warning(f"Exemple {i} incomplet, clés manquantes. Clés disponibles: {example.keys()}")
                    continue
                    
                input_text = example[input_field]
                reference = example[output_field]
                
                # Vérifier que les textes ne sont pas vides
                if not input_text or not reference:
                    logger.warning(f"Exemple {i} avec texte vide: input={bool(input_text)}, output={bool(reference)}")
                    continue
                    
                # Générer la prédiction
                logger.debug(f"Génération pour l'exemple {i}: {input_text[:50]}...")
                generated_text = self.generator.generate(input_text)
                
                # Enregistrer les résultats
                results.append({
                    "input": input_text,
                    "reference": reference,
                    "generated": generated_text
                })
                metrics_data["predictions"].append(generated_text)
                metrics_data["references"].append(reference)
                valid_examples += 1
                
            except Exception as e:
                logger.error(f"Erreur lors du traitement de l'exemple {i}: {e}")
                continue
        
        logger.info(f"Validation terminée: {valid_examples}/{len(examples_to_process)} exemples valides")
        
        # Vérifier si des exemples valides ont été trouvés
        if not metrics_data["predictions"]:
            logger.error("Aucun exemple valide n'a été trouvé pour la validation.")
            return {"examples": results, "metrics": {}, "error": "Aucun exemple valide"}
        
        # Calculer les métriques
        try:
            metrics_scores = self.metrics.compute_metrics(
                metrics_data["predictions"], 
                metrics_data["references"]
            )
            logger.info(f"Métriques calculées: {metrics_scores}")
        except Exception as e:
            logger.error(f"Erreur lors du calcul des métriques: {e}")
            metrics_scores = {"error": str(e)}
        
        return {
            "examples": results,
            "metrics": metrics_scores,
            "num_valid": valid_examples
        }

def get_tokenizer(model_name: str):
    """Configure le tokenizer avec les paramètres optimaux."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            model_max_length=512,
            padding_side="right",
            truncation_side="right"
        )
        logger.info(f"Tokenizer chargé avec succès: {model_name}")
        return tokenizer
    except Exception as e:
        logger.error(f"Erreur lors du chargement du tokenizer {model_name}: {e}")
        raise

def preprocess_function(examples, tokenizer, max_input_length, max_target_length):
    try:
        # Nettoyage amélioré des données
        inputs = [
            " ".join(str(text).strip().split())
            for text in examples["input"]
            if text and str(text).strip()
        ]
        outputs = [
            " ".join(str(text).strip().split())
            for text in examples["output"]
            if text and str(text).strip()
        ]
        
        # Tokenisation avec gestion de la longueur
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
            return_attention_mask=True
        )
        
        # Tokenisation des sorties
        labels = tokenizer(
            outputs,
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
            return_attention_mask=True
        )
        
        # Optimisation des labels avec gestion du padding
        model_inputs["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token
             for token in label]
            for label in labels["input_ids"]
        ]
        
        return model_inputs
    except Exception as e:
        logger.error(f"Erreur de prétraitement: {e}")
        raise

def verify_jsonl_file(file_path: str) -> bool:
    """Vérifie la validité d'un fichier JSONL."""
    if not os.path.exists(file_path):
        logger.error(f"Fichier introuvable: {file_path}")
        return False
        
    try:
        valid_lines = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and "input" in data and "output" in data:
                        if data["input"] and data["output"]:
                            valid_lines += 1
                except json.JSONDecodeError:
                    logger.warning(f"Ligne {i} n'est pas un JSON valide")
                    
        logger.info(f"Vérification du fichier {file_path}: {valid_lines} exemples valides")
        return valid_lines > 0
    except Exception as e:
        logger.error(f"Erreur lors de la vérification du fichier {file_path}: {e}")
        return False

@safe_execute
def train_mt5_model(
    train_path: str,
    val_path: Optional[str] = None,
    model_name: str = "google/mt5-small",
    output_dir: str = "models/mt5_antandroy",
    batch_size: int = 8,          # Augmenté
    epochs: int = 5,              # Augmenté
    lr: float = 2e-4,            # Optimisé
    max_input_length: int = 128,  # Augmenté
    max_target_length: int = 128, # Augmenté
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,    # Changé de steps à ratio
    gradient_accumulation_steps: int = 4,  # Augmenté
    eval_steps: int = 500,        # Augmenté
    seed: int = 42,
    fp16: bool = True,           # Activé par défaut
    resume_from_checkpoint: bool = True
) -> Dict[str, Any]:
    """Entraîne un modèle MT5 sur un dataset en malgache."""
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Utilisation du dispositif: {device}")
    
    # Fixer la graine pour la reproductibilité
    set_seed(seed)
    logger.info(f"Graine aléatoire fixée à {seed}")
    
    # Créer les répertoires nécessaires
    os.makedirs(output_dir, exist_ok=True)
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Vérifier les fichiers d'entrée
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Fichier d'entraînement introuvable: {train_path}")
        
    if val_path and not os.path.exists(val_path):
        logger.warning(f"Fichier de validation introuvable: {val_path}. Une partie des données d'entraînement sera utilisée pour la validation.")
        val_path = None
    
    # Vérifier la validité des fichiers JSONL
    logger.info("Vérification des fichiers de données...")
    if train_path.endswith('.jsonl') and not verify_jsonl_file(train_path):
        logger.warning(f"Le fichier d'entraînement {train_path} contient des problèmes de format.")
    
    if val_path and val_path.endswith('.jsonl') and not verify_jsonl_file(val_path):
        logger.warning(f"Le fichier de validation {val_path} contient des problèmes de format.")
    
    # Charger les données
    logger.info(f"Chargement des données depuis {train_path}...")
    try:
        dataset = DatasetDict({
            "train": load_dataset("json", data_files={"train": train_path})["train"],
            "validation": (
                load_dataset("json", data_files={"validation": val_path})["validation"]
                if val_path and os.path.exists(val_path)
                else load_dataset("json", data_files={"train": train_path})["train"].train_test_split(
                    test_size=0.1, seed=seed
                )["test"]
            )
        })
        
        logger.info(f"Données chargées: {len(dataset['train'])} exemples d'entraînement, {len(dataset['validation'])} exemples de validation")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        raise
    
    # Télécharger le tokenizer et le modèle
    try:
        logger.info(f"Chargement du tokenizer et du modèle {model_name}...")
        tokenizer = get_tokenizer(model_name)
        model = MT5ForConditionalGeneration.from_pretrained(model_name).to(device)
        logger.info("Modèle chargé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        raise
    
    # Prétraiter les données
    logger.info("Prétraitement des données...")
    try:
        tokenized_datasets = dataset.map(
            lambda x: preprocess_function(x, tokenizer, max_input_length, max_target_length),
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenisation des données",
            num_proc=4
        )
        logger.info("Prétraitement terminé")
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement des données: {e}")
        raise

    # Configuration de l'entraînement
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=weight_decay,
        num_train_epochs=epochs,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=fp16 and torch.cuda.is_available(),
        eval_steps=eval_steps,
        save_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=logs_dir,
        logging_steps=100,
        save_safetensors=True,
        gradient_checkpointing=True,
        report_to="tensorboard",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        group_by_length=True,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        remove_unused_columns=True,
        label_smoothing_factor=0.1
    )
    # Création du trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding="max_length"
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Vérifier s'il existe un checkpoint
    last_checkpoint = None
    if resume_from_checkpoint:
        last_checkpoint = get_last_checkpoint_safely(output_dir)
        if last_checkpoint:
            logger.info(f"Reprise de l'entraînement depuis {last_checkpoint}")
        else:
            logger.info("Aucun checkpoint valide trouvé, démarrage d'un nouvel entraînement")
    
    # Lancer l'entraînement
    logger.info("Début de l'entraînement...")
    try:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Entraînement terminé: {train_result.metrics}")
        
        # Sauvegarder les métriques d'entraînement
        with open(os.path.join(output_dir, "train_results.json"), "w") as f:
            json.dump(train_result.metrics, f, indent=2)
    except Exception as e:
        logger.error(f"Erreur pendant l'entraînement: {e}")
        raise

    # Évaluation finale
    logger.info("Évaluation du modèle final...")
    metrics = CustomMetrics()
    validator = ModelValidator(model, tokenizer, device, metrics)
    
    # Valider sur l'ensemble de validation
    validation_results = validator.validate(
        dataset["validation"],
        num_examples=min(5, len(dataset["validation"]))
    )
    
    # Compiler et sauvegarder les résultats
    training_duration = time.time() - start_time
    hours, remainder = divmod(training_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    results = {
        "model_path": output_dir,
        "training_time": f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
        "training_time_seconds": training_duration,
        "metrics": validation_results["metrics"],
        "examples": validation_results["examples"],
        "model_config": {
            "model_name": model_name,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "max_input_length": max_input_length,
            "max_target_length": max_target_length,
            "device": device,
            "fp16": fp16 and torch.cuda.is_available(),
            "transformers_version": transformers_version
        }
    }
    
    # Sauvegarder les résultats
    with open(os.path.join(output_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Entraînement terminé en {results['training_time']}")
    return results

def get_last_checkpoint_safely(output_dir: str) -> Optional[str]:
    """Récupère le dernier checkpoint de manière sécurisée."""
    try:
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint and os.path.exists(os.path.join(last_checkpoint, "trainer_state.json")):
            logger.info(f"Checkpoint trouvé: {last_checkpoint}")
            return last_checkpoint
        
        # Recherche manuelle si la méthode standard échoue
        checkpoints = [
            os.path.join(output_dir, d) 
            for d in os.listdir(output_dir) 
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ]
        
        if checkpoints:
            # Trier par numéro de checkpoint (plus récent en premier)
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)
            for checkpoint in checkpoints:
                if os.path.exists(os.path.join(checkpoint, "trainer_state.json")):
                    logger.info(f"Checkpoint trouvé manuellement: {checkpoint}")
                    return checkpoint
                    
        return None
    except Exception as e:
        logger.warning(f"Erreur lors de la récupération du dernier checkpoint: {str(e)}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning MT5 pour la traduction malgache-antandroy")
    parser.add_argument("--train", required=True, help="Chemin du fichier d'entraînement (JSONL)")
    parser.add_argument("--validation", help="Chemin du fichier de validation (JSONL)")
    parser.add_argument("--model", default="google/mt5-small", help="Nom du modèle pré-entraîné")
    parser.add_argument("--output", default="models/mt5_antandroy", help="Répertoire de sortie")
    parser.add_argument("--batch-size", type=int, default=4, help="Taille des batchs")
    parser.add_argument("--epochs", type=int, default=3, help="Nombre d'époques")
    parser.add_argument("--lr", type=float, default=5e-5, help="Taux d'apprentissage")
    parser.add_argument("--max-input-length", type=int, default=64, help="Longueur maximale de l'entrée")
    parser.add_argument("--max-target-length", type=int, default=64, help="Longueur maximale de la sortie")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Étapes de warmup")
    parser.add_argument("--eval-steps", type=int, default=100, help="Fréquence d'évaluation")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire")
    parser.add_argument("--fp16", action="store_true", help="Utiliser la précision mixte (FP16)")
    parser.add_argument("--verify-data", action="store_true", help="Vérifier la validité des données avant l'entraînement")
    parser.add_argument("--no-resume", action="store_true", help="Ne pas reprendre depuis le dernier checkpoint")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Ratio de warmup")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Factor de label smoothing")
    args = parser.parse_args()

    # Vérification des données si demandé
    if args.verify_data:
        logger.info("Vérification des données d'entraînement...")
        train_valid = verify_jsonl_file(args.train)
        if not train_valid:
            logger.error("Les données d'entraînement présentent des problèmes. Vérifiez le format JSONL.")
            exit(1)
            
        if args.validation:
            logger.info("Vérification des données de validation...")
            val_valid = verify_jsonl_file(args.validation)
            if not val_valid:
                logger.warning("Les données de validation présentent des problèmes. Une partie des données d'entraînement sera utilisée à la place.")
                args.validation = None

    # Configuration système avant l'entraînement
    logger.info("Configuration du système...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"CUDA disponible: {torch.cuda.get_device_name(0)}")
        logger.info(f"Mémoire CUDA: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("CUDA non disponible, l'entraînement sera effectué sur CPU (beaucoup plus lent)")

    # Lancer l'entraînement
    logger.info("Démarrage de l'entraînement...")
    try:
        results = train_mt5_model(
            train_path=args.train,
            val_path=args.validation,
            model_name=args.model,
            output_dir=args.output,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length,
            warmup_ratio=args.warmup_ratio,
            eval_steps=args.eval_steps,
            seed=args.seed,
            fp16=True,  # Activé par défaut pour de meilleures performances
            resume_from_checkpoint=not args.no_resume
        )
        
        logger.info(f"Entraînement terminé avec succès. Modèle sauvegardé dans: {args.output}")
        logger.info(f"Métriques finales: {results['metrics']}")
        
        # Afficher quelques exemples de traduction
        logger.info("Exemples de traduction:")
        for i, example in enumerate(results['examples'][:3], 1):
            logger.info(f"Exemple {i}:")
            logger.info(f"  Source:    {example['input']}")
            logger.info(f"  Référence: {example['reference']}")
            logger.info(f"  Généré:    {example['generated']}")
            
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {e}")
        logger.error(traceback.format_exc())
        exit(1)