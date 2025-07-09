import torch
import json
import argparse
from scripts.train_lstm import Seq2SeqWithAttention, ImprovedTokenizer

def load_tokenizer(vocab_path):
    """
    Charge le vocabulaire et crée le tokenizer.
    """
    tokenizer = ImprovedTokenizer(vocab_file=vocab_path)
    return tokenizer

def load_model(model_path, vocab_size, embedding_dim, hidden_dim, n_layers, dropout, device):
    """
    Initialise le modèle, charge les poids sauvegardés et le positionne en mode évaluation.
    """
    model = Seq2SeqWithAttention(vocab_size, embedding_dim, hidden_dim, n_layers, dropout)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test interactif avec le modèle Seq2Seq LSTM avec attention")
    parser.add_argument("--model-dir", default="models/lstm_seq2seq", help="Répertoire du modèle sauvegardé")
    parser.add_argument("--embedding-dim", type=int, default=256, help="Dimension des embeddings")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Dimension des états cachés")
    parser.add_argument("--layers", type=int, default=2, help="Nombre de couches LSTM")
    parser.add_argument("--dropout", type=float, default=0.3, help="Taux de dropout")
    # Vous pouvez ajouter ultérieurement un argument pour choisir un décodage beam (si implémenté)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = f"{args.model_dir}/final_model.pth"
    vocab_path = f"{args.model_dir}/vocab.json"
    
    # Charger le tokenizer et obtenir la taille du vocabulaire
    tokenizer = load_tokenizer(vocab_path)
    vocab_size = tokenizer.vocab_size()
    
    # Charger le modèle
    model = load_model(model_path, vocab_size, args.embedding_dim, args.hidden_dim, args.layers, args.dropout, device)
    
    print("Tapez votre message ('exit' pour quitter) :")
    while True:
        user_input = input("Vous: ")
        if user_input.lower() == "exit":
            break
        
        # Encoder l'entrée
        input_indices = tokenizer.encode(user_input, max_len=64)
        input_tensor = torch.tensor(input_indices).unsqueeze(0).to(device)  # [1, seq_len]
        
        # Générer la réponse (ici décodage greedy)
        generated = model.generate(input_tensor, max_len=32)
        response = tokenizer.decode(generated[0].cpu().numpy())
        
        print("Modèle:", response)