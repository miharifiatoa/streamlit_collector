import os
import argparse
import logging
import time
import json
import random
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ImprovedTokenizer:
    """Tokenizer simple mais efficace avec sous-échantillonnage pour les grands corpus"""
    
    def __init__(self, texts=None, vocab_file=None, max_vocab_size=10000):
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<bos>", 3: "<eos>"}
        self.max_vocab_size = max_vocab_size
        self.vocab = {}

        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)
        elif texts:
            self.build_vocab(texts)

    def build_vocab(self, texts):
        logger.info(f"Construction du vocabulaire (max {self.max_vocab_size} tokens)...")
        if len(texts) > 10000:
            logger.info(f"Sous-échantillonnage de {len(texts)} textes à 10000 pour la construction du vocabulaire")
            random.seed(42)
            texts = random.sample(texts, 10000)

        # Compter les occurrences des mots
        for text in tqdm(texts, desc="Comptage des mots"):
            if isinstance(text, str):  # Si le texte est une chaîne
                words = text.strip().split()
            elif isinstance(text, list):  # Si le texte est déjà une liste de tokens
                words = text
            else:
                continue
                
            for word in words:
                self.vocab[word] = self.vocab.get(word, 0) + 1

        # Construire word2idx et idx2word avec les mots les plus fréquents
        sorted_words = sorted(self.vocab.items(), key=lambda x: x[1], reverse=True)
        sorted_words = sorted_words[:self.max_vocab_size - 4]  # -4 pour les tokens spéciaux
        
        for i, (word, _) in enumerate(sorted_words):
            self.word2idx[word] = i + 4
            self.idx2word[i + 4] = word

        logger.info(f"Vocabulaire construit: {len(self.word2idx)} tokens")

    def save_vocab(self, vocab_file):
        logger.info(f"Sauvegarde du vocabulaire dans {vocab_file}")
        os.makedirs(os.path.dirname(vocab_file), exist_ok=True)
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=2)

    def load_vocab(self, vocab_file):
        logger.info(f"Chargement du vocabulaire depuis {vocab_file}")
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.word2idx = json.load(f)
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        logger.info(f"Vocabulaire chargé: {len(self.word2idx)} tokens")

    def encode(self, text, max_len=None):
        if isinstance(text, list):
            words = text
        else:
            words = text.strip().split()
            
        tokens = [2]  # <bos>
        for word in words:
            tokens.append(self.word2idx.get(word, 1))  # 1 est l'index de <unk>
        tokens.append(3)  # <eos>
        
        if max_len and len(tokens) > max_len:
            return tokens[:max_len-1] + [3]
        return tokens

    def decode(self, indices):
        tokens = []
        for idx in indices:
            if idx == 3:  # <eos>
                break
            if idx > 3:
                tokens.append(self.idx2word.get(idx, "<unk>"))
        return " ".join(tokens)

    def vocab_size(self):
        return len(self.word2idx)

class DataAugmentation:
    @staticmethod
    def add_noise(text, p=0.1):
        """Ajoute du bruit aléatoire au texte"""
        words = text.split()
        for i in range(len(words)):
            if random.random() < p:
                # Ajoute, supprime ou remplace un caractère aléatoire
                if len(words[i]) > 3:
                    op = random.choice(['add', 'delete', 'replace'])
                    if op == 'add':
                        pos = random.randint(0, len(words[i]))
                        char = random.choice('abcdefghijklmnopqrstuvwxyz')
                        words[i] = words[i][:pos] + char + words[i][pos:]
                    elif op == 'delete':
                        pos = random.randint(0, len(words[i])-1)
                        words[i] = words[i][:pos] + words[i][pos+1:]
                    else:  # replace
                        pos = random.randint(0, len(words[i])-1)
                        char = random.choice('abcdefghijklmnopqrstuvwxyz')
                        words[i] = words[i][:pos] + char + words[i][pos+1:]
        return ' '.join(words)

    @staticmethod
    def shuffle_words(text, p=0.1):
        """Mélange légèrement l'ordre des mots"""
        words = text.split()
        if len(words) > 2 and random.random() < p:
            i = random.randint(0, len(words)-2)
            words[i], words[i+1] = words[i+1], words[i]
        return ' '.join(words)

class AdvancedChatDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_len=64, train_val_split=None, is_train=True, seed=42, augment=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pairs = []
        self.augment = augment and is_train
        self.data_augmenter = DataAugmentation()
        
        logger.info(f"Chargement du dataset depuis {json_file}")
        try:
            with open(json_file, encoding='utf-8') as f:
                data = json.load(f)
                
                # Vérifier que data est un dictionnaire
                if not isinstance(data, dict):
                    raise ValueError("Le fichier JSON doit contenir un objet")

                # Récupérer les entrées et les exemples
                inputs = data.get("input", [])
                examples = data.get("exemples", [])  # Utiliser "exemples" au lieu de "output"

                # Vérifier que inputs est une liste non vide
                if not isinstance(inputs, list) or not inputs:
                    raise ValueError("La clé 'input' doit contenir une liste non vide")

                # Si examples est vide, utiliser la liste des inputs comme exemples
                if not examples:
                    examples = inputs

                # Filtrer les lignes vides
                inputs = [x.strip() for x in inputs if isinstance(x, str) and x.strip()]
                examples = [x.strip() for x in examples if isinstance(x, str) and x.strip()]

                # Création des paires input-exemple
                for input_text, example_text in zip(inputs, examples):
                    if input_text and example_text:
                        # Données originales
                        src = torch.tensor(self.tokenizer.encode(input_text, self.max_len))
                        tgt = torch.tensor(self.tokenizer.encode(example_text, self.max_len))
                        self.pairs.append((src, tgt))
                        
                        # Augmentation des données pour l'ensemble d'entraînement
                        if self.augment:
                            # Version avec bruit
                            noisy_input = self.data_augmenter.add_noise(input_text)
                            src_noisy = torch.tensor(self.tokenizer.encode(noisy_input, self.max_len))
                            self.pairs.append((src_noisy, tgt))
                            
                            # Version avec mots mélangés
                            shuffled_input = self.data_augmenter.shuffle_words(input_text)
                            src_shuffled = torch.tensor(self.tokenizer.encode(shuffled_input, self.max_len))
                            self.pairs.append((src_shuffled, tgt))

        except json.JSONDecodeError as e:
            logger.error(f"Erreur de décodage JSON: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Erreur lors du chargement du fichier JSON: {str(e)}")
            raise

        if not self.pairs:
            raise ValueError("Aucune paire de données valide trouvée dans le fichier JSON")

        # Split train/val si nécessaire
        if train_val_split is not None:
            random.seed(seed)
            random.shuffle(self.pairs)
            split_idx = int(len(self.pairs) * (1 - train_val_split))
            self.pairs = self.pairs[:split_idx] if is_train else self.pairs[split_idx:]

        # Logging des statistiques
        src_lens = [len(src) for src, _ in self.pairs]
        tgt_lens = [len(tgt) for _, tgt in self.pairs]
        
        logger.info(f"Dataset {'train' if is_train else 'val'}: {len(self.pairs)} exemples")
        if src_lens:
            logger.info(f"Longueur moyenne source: {sum(src_lens)/len(src_lens):.1f} tokens")
            logger.info(f"Longueur moyenne cible: {sum(tgt_lens)/len(tgt_lens):.1f} tokens")
            logger.info(f"Longueur maximale source: {max(src_lens)} tokens")
            logger.info(f"Longueur maximale cible: {max(tgt_lens)} tokens")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch

class EncoderLSTM(nn.Module):
    """Encodeur LSTM avec embeddings et dropout"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=1, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=n_layers, 
            batch_first=True, 
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim*2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.fc(hidden).unsqueeze(0)
        cell = torch.cat([cell[-2], cell[-1]], dim=1)
        cell = self.fc(cell).unsqueeze(0)
        return outputs, (hidden, cell)

class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=1, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = nn.Linear(hidden_dim*2 + hidden_dim + embedding_dim, 1)
        self.lstm = nn.LSTM(
            embedding_dim + hidden_dim*2,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_dim + hidden_dim*2 + embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def attention_mechanism(self, hidden, encoder_outputs, decoder_input):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        hidden_last = hidden[-1].unsqueeze(1)
        hidden_repeated = hidden_last.repeat(1, src_len, 1)
        embedded = self.embedding(decoder_input)
        embedded_repeated = embedded.repeat(1, src_len, 1)
        energy_input = torch.cat((hidden_repeated, encoder_outputs, embedded_repeated), dim=2)
        energy = self.attention(energy_input)
        attention = torch.softmax(energy.squeeze(-1), dim=1).unsqueeze(1)
        context = torch.bmm(attention, encoder_outputs)
        return context, attention

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1) if input.dim() == 1 else input
        embedded = self.dropout(self.embedding(input))
        context, attention = self.attention_mechanism(hidden, encoder_outputs, input)
        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output_context = torch.cat((output, context, embedded), dim=2)
        prediction = self.fc_out(output_context.squeeze(1))
        return prediction, hidden, cell, attention

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=1, dropout=0.2):
        super().__init__()
        self.encoder = EncoderLSTM(vocab_size, embedding_dim, hidden_dim, n_layers, dropout)
        self.decoder = AttentionDecoder(vocab_size, embedding_dim, hidden_dim, n_layers, dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_layers = n_layers

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = tgt.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, tgt_len-1, tgt_vocab_size).to(self.device)
        encoder_outputs, (hidden, cell) = self.encoder(src)
        hidden = hidden.repeat(self.n_layers, 1, 1)
        cell = cell.repeat(self.n_layers, 1, 1)
        decoder_input = tgt[:, 0]  # <bos>
        for t in range(1, tgt_len):
            output, hidden, cell, _ = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs[:, t-1] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = tgt[:, t] if teacher_force else top1
        return outputs

    def generate(self, src, max_len=64):
        batch_size = src.shape[0]
        encoder_outputs, (hidden, cell) = self.encoder(src)
        hidden = hidden.repeat(self.n_layers, 1, 1)
        cell = cell.repeat(self.n_layers, 1, 1)
        decoder_input = torch.full((batch_size,), 2, dtype=torch.long, device=self.device)
        generated_tokens = []
        for t in range(max_len):
            output, hidden, cell, _ = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            top1 = output.argmax(1)
            generated_tokens.append(top1.unsqueeze(1))
            decoder_input = top1
        generated = torch.cat(generated_tokens, dim=1)
        return generated

def train_model(
    data_path,
    model_dir="models/lstm_seq2seq",
    vocab_file=None,
    embedding_dim=128,
    hidden_dim=256,
    n_layers=1,
    batch_size=32,
    epochs=10,
    learning_rate=0.001,
    max_len=128,
    dropout=0.5,
    clip=1.0,
    seed=42,
    val_split=0.1,
    weight_decay=1e-5
):
    start_time = time.time()
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Utilisation du device: {device}")
    logger.info("Préparation des données...")
    if not vocab_file or not os.path.exists(vocab_file):
        logger.info("Construction du vocabulaire...")
        all_texts = []
        with open(data_path, encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    # Si la liste se trouve sous une clé par exemple "data"
                    data = data.get("data", [data])
                for item in data:
                    all_texts.append(item["input"])
                    all_texts.append(item.get("exemple", ""))
            except Exception as e:
                logger.error(f"Erreur lors du chargement du vocabulaire depuis le fichier JSON: {e}")
        tokenizer = ImprovedTokenizer(all_texts)
        if vocab_file:
            tokenizer.save_vocab(vocab_file)
    else:
        tokenizer = ImprovedTokenizer(vocab_file=vocab_file)
    train_dataset = AdvancedChatDataset(
        data_path, 
        tokenizer, 
        max_len=max_len, 
        train_val_split=val_split, 
        is_train=True, 
        seed=seed,
        augment=True
    )
    val_dataset = AdvancedChatDataset(
        data_path, 
        tokenizer, 
        max_len=max_len, 
        train_val_split=val_split, 
        is_train=False, 
        seed=seed,
        augment=False
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    logger.info("Initialisation du modèle...")
    model = Seq2SeqWithAttention(
        vocab_size=tokenizer.vocab_size(),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Nombre de paramètres: {n_params:,}")
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), 
                          lr=learning_rate,
                          weight_decay=weight_decay)  
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    logger.info(f"Démarrage de l'entraînement sur {epochs} époques...")
    for epoch in range(epochs):
        start_epoch = time.time()
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for src, tgt in pbar:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt, teacher_forcing_ratio=0.5)
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)
            loss = criterion(output, tgt)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        model.eval()
        epoch_loss = 0
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for src, tgt in pbar:
                src, tgt = src.to(device), tgt.to(device)
                output = model(src, tgt, teacher_forcing_ratio=0.0)
                output_dim = output.shape[-1]
                output = output.reshape(-1, output_dim)
                tgt = tgt[:, 1:].reshape(-1)
                loss = criterion(output, tgt)
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        val_loss = epoch_loss / len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        epoch_mins, epoch_secs = divmod(time.time() - start_epoch, 60)
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Temps: {epoch_mins:.0f}m {epoch_secs:.0f}s")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'vocab_size': tokenizer.vocab_size(),
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'n_layers': n_layers,
                'dropout': dropout
            }, os.path.join(model_dir, 'best_model.pth'))
            logger.info(f"Meilleur modèle sauvegardé!")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'vocab_size': tokenizer.vocab_size(),
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'n_layers': n_layers,
        'dropout': dropout
    }, os.path.join(model_dir, 'final_model.pth'))
    tokenizer.save_vocab(os.path.join(model_dir, 'vocab.json'))
    test_samples = []
    for src, _ in val_dataset:
        if len(test_samples) < 5:
            test_samples.append(src)
    logger.info("Exemples de génération:")
    model.eval()
    with torch.no_grad():
        for src in test_samples:
            src = src.unsqueeze(0).to(device)
            generated = model.generate(src)
            src_text = tokenizer.decode(src[0].cpu().numpy())
            gen_text = tokenizer.decode(generated[0].cpu().numpy())
            logger.info(f"Entrée: {src_text}")
            logger.info(f"Génération: {gen_text}")
            logger.info("-" * 50)
    total_time = time.time() - start_time
    logger.info(f"✅ Entraînement terminé en {total_time/60:.2f} minutes")
    logger.info(f"Meilleure perte de validation: {best_val_loss:.4f}")
    logger.info(f"Modèle sauvegardé dans {model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne un modèle Seq2Seq LSTM avec attention")
    parser.add_argument("--data", default="data/processed/output.json", 
                        help="Chemin vers le fichier de données JSONL")
    parser.add_argument("--model-dir", default="models/lstm_seq2seq", 
                        help="Répertoire pour sauvegarder le modèle")
    parser.add_argument("--vocab", default=None, 
                        help="Chemin vers un fichier vocabulaire existant (optionnel)")
    parser.add_argument("--embedding-dim", type=int, default=128, 
                        help="Dimension des embeddings")
    parser.add_argument("--hidden-dim", type=int, default=256, 
                        help="Dimension des états cachés LSTM")
    parser.add_argument("--layers", type=int, default=2, 
                        help="Nombre de couches LSTM")
    parser.add_argument("--batch-size", type=int, default=64, 
                        help="Taille des mini-batches")
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Nombre d'époques d'entraînement")
    parser.add_argument("--lr", type=float, default=0.01, 
                        help="Taux d'apprentissage")
    parser.add_argument("--max-len", type=int, default=128, 
                        help="Longueur maximale des séquences")
    parser.add_argument("--dropout", type=float, default=0.3, 
                        help="Taux de dropout")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Graine pour la reproductibilité")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Coefficient de régularisation L2")
    args = parser.parse_args()
    
    train_model(
        data_path=args.data,
        model_dir=args.model_dir,
        vocab_file=args.vocab,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        max_len=args.max_len,
        dropout=args.dropout,
        seed=args.seed,
         weight_decay=args.weight_decay
    )