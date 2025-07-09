import streamlit as st
import os
import json
import asyncio
import nest_asyncio
import torch
import time
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from transformers import MT5ForConditionalGeneration, AutoTokenizer
import importlib.util
import sys
from audio_recorder_streamlit import audio_recorder

# Configuration Streamlit (doit être la première commande Streamlit)
st.set_page_config(
    page_title="Chatbot Antandroy",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation de l'état de session
if 'page_active' not in st.session_state:
    st.session_state.page_active = "💬 Chatbot"
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'history' not in st.session_state:
    st.session_state.history = []

# Définition du chemin du script de prétraitement
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESS_PATH = os.path.join(SCRIPT_DIR, "pretraitement.py")

# Script de prétraitement par défaut si introuvable
default_preprocess = """
import re
import json
from typing import List, Dict

def process_text(text: str) -> Dict[str, List[str]]:
    # Extraction des mots uniquement alphabétiques en minuscules
    words = [word for word in re.findall(r'\\b\\w+\\b', text.lower()) if word.isalpha()]
    unique_words = list(dict.fromkeys(words))
    return {
        "input": unique_words,
        "output": [""] * len(unique_words),
        "examples": [""] * len(unique_words)
    }

def save_to_json(data: Dict, output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
"""

# Vérifie si le fichier de prétraitement existe, sinon le crée
if not os.path.exists(PREPROCESS_PATH):
    st.info("Création d'un fichier de prétraitement par défaut...")
    with open(PREPROCESS_PATH, 'w', encoding='utf-8') as f:
        f.write(default_preprocess)

# Import dynamique du module de prétraitement
try:
    spec = importlib.util.spec_from_file_location("pretraitement", PREPROCESS_PATH)
    pretraitement = importlib.util.module_from_spec(spec)
    sys.modules["pretraitement"] = pretraitement
    spec.loader.exec_module(pretraitement)
    process_text = pretraitement.process_text
    save_to_json = pretraitement.save_to_json
except Exception as e:
    st.error(f"❌ Erreur lors du chargement du prétraitement: {str(e)}")
    st.stop()

# Configuration initiale
os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
try:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
except Exception as e:
    st.error(f"Erreur d'initialisation asyncio: {str(e)}")

# Constants
MODEL_PATH = "models/mt5_antandroy"
DATASET_PATH = "data/processed/output.json"
DEFAULT_MAX_LENGTH = 256
DEFAULT_TEMPERATURE = 0.7

# Thème sombre personnalisé
st.markdown("""
    <style>
    body { background-color: #1a1a1a; color: #e0e0e0; }
    .css-1d391kg { background-color: #333333; color: #e0e0e0; }
    .stButton button {
        background-color: #4a90e2;
        color: white;
        border-radius: 0.3rem;
        border: none;
    }
    .stButton button:hover { background-color: #357ABD; }
    input, textarea {
        border: 1px solid #555555 !important;
        border-radius: 0.3rem !important;
        background-color: #2a2a2a !important;
        color: #e0e0e0 !important;
    }
    .stHorizontalBlock {
        display: flex;
        justify-content: flex-start;
        padding: 0.5rem;
        background-color: #3d3d3d;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stHorizontalBlock"] > div:first-child { flex: 0 1 auto; }
    </style>
""", unsafe_allow_html=True)

# Classe gérant le modèle MT5
class ModelManager:
    @st.cache_resource
    def load_model() -> Tuple[Optional[MT5ForConditionalGeneration], Optional[AutoTokenizer]]:
        try:
            with st.spinner("Chargement du modèle en cours..."):
                model = MT5ForConditionalGeneration.from_pretrained(MODEL_PATH)
                tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
                return model, tokenizer
        except RuntimeError as e:
            if "torch.classes" in str(e):
                st.warning("⚠️ Avertissement: Problème de chargement des classes torch. Le modèle continuera à fonctionner.")
                return None, None
            st.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
            return None, None
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
            return None, None

    @staticmethod
    def generate_response(
        input_text: str,
        model: MT5ForConditionalGeneration,
        tokenizer: AutoTokenizer,
        max_length: int = DEFAULT_MAX_LENGTH,
        temperature: float = DEFAULT_TEMPERATURE
    ) -> Tuple[str, float]:
        start_time = time.time()
        try:
            inputs = tokenizer.encode(input_text, return_tensors="pt")
            outputs = model.generate(
                inputs,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return decoded, time.time() - start_time
        except Exception as e:
            return f"Erreur de génération: {str(e)}", 0

# Classe gérant le dataset (chargement, sauvegarde, statistiques)
class DatasetManager:
    @st.cache_resource
    def load_dataset() -> Dict:
        try:
            with open(DATASET_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Erreur lors du chargement du dataset: {str(e)}")
            return {"input": [], "output": [], "examples": []}

    @staticmethod
    def save_dataset(data: Dict) -> bool:
        try:
            with open(DATASET_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            st.error(f"Erreur lors de la sauvegarde: {str(e)}")
            return False

    @staticmethod
    def get_statistics(dataset: Dict) -> Dict:
        total_words = len(dataset['input'])
        translated_words = len([x for x in dataset['output'] if x])
        return {
            "total": total_words,
            "translated": translated_words,
            "progress": (translated_words / total_words * 100) if total_words > 0 else 0
        }

# Classe d'édition du dataset
class DatasetEditor:
    def __init__(self):
        self.dataset = DatasetManager.load_dataset()
        if not self.dataset["input"]:
            self.dataset = {
                "input": ["Exemple"],
                "output": [""],
                "examples": [""]
            }
            DatasetManager.save_dataset(self.dataset)
        self.stats = DatasetManager.get_statistics(self.dataset)

    def render_navigation(self, current_index: int) -> int:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Précédent", disabled=current_index <= 0):
                return current_index - 1
        with col2:
            st.write(f"Mot {current_index + 1}/{self.stats['total']}")
        with col3:
            if st.button("Suivant ➡️", disabled=current_index >= self.stats['total'] - 1):
                return current_index + 1
        return current_index

    def render_sidebar(self):
        st.sidebar.subheader("📊 Statistiques")
        st.sidebar.info(f"Total des mots: {self.stats['total']}")
        st.sidebar.info(f"Mots traduits: {self.stats['translated']}")
        st.sidebar.progress(self.stats['progress'] / 100, f"Progression: {self.stats['progress']:.1f}%")

    def render_voice_recorder(self, current_index: int):
        st.markdown("### 🎤 Enregistrez votre voix pour ce mot ou cette phrase")
        audio_dir = "data/voices"
        os.makedirs(audio_dir, exist_ok=True)
        audio_bytes = audio_recorder(
            text="Cliquez sur le micro pour enregistrer/arrêter",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=2.0,
            sample_rate=41000,
        )
        if audio_bytes:
            base_name = self.dataset['input'][current_index].replace(" ", "_")
            save_path = os.path.join(audio_dir, f"{base_name}_{current_index}.wav")
            with open(save_path, "wb") as f:
                f.write(audio_bytes)
            st.success(f"✅ Enregistrement sauvegardé sous {save_path}")
            st.audio(audio_bytes, format="audio/wav")

    def render_edit_form(self, current_index: int) -> int:
        st.subheader("✏️ Édition du mot")
        with st.form(key=f"edit_form_{current_index}"):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### Mot source")
                input_word = st.text_input(
                    "Mot en Antandroy",
                    value=self.dataset['input'][current_index],
                    key=f"input_{current_index}"
                )
                st.markdown("### Traduction")
                output_word = st.text_input(
                    "Traduction en français",
                    value=self.dataset['output'][current_index] if current_index < len(self.dataset['output']) else "",
                    key=f"output_{current_index}"
                )
            with col2:
                st.markdown("### Exemple d'utilisation")
                example = st.text_area(
                    "Phrase exemple",
                    value=self.dataset['examples'][current_index] if 'examples' in self.dataset and current_index < len(self.dataset['examples']) else "",
                    height=150,
                    key=f"example_{current_index}",
                    help="Entrez une phrase d'exemple utilisant ce mot"
                )
            st.markdown("---")
            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
            with btn_col1:
                submit = st.form_submit_button("💾 Sauvegarder", use_container_width=True, type="primary")
            with btn_col2:
                skip = st.form_submit_button("⏭️ Passer", use_container_width=True)
            with btn_col3:
                st.caption("Position actuelle:")
                st.progress((current_index + 1) / self.stats['total'])
            if submit:
                if self.save_word(current_index, input_word, output_word, example):
                    st.success("✅ Mot sauvegardé!")
                    return current_index + 1
            elif skip:
                return current_index + 1
        with st.expander("ℹ️ Informations du mot", expanded=False):
            st.markdown(f"""
            **Index**: {current_index + 1} / {self.stats['total']}  
            **Statut**: {'✅ Traduit' if self.dataset['output'][current_index] else '❌ Non traduit'}  
            **Exemple**: {'✅ Présent' if 'examples' in self.dataset and self.dataset['examples'][current_index] else '❌ Absent'}
            """)
        self.render_voice_recorder(current_index)
        return current_index

    def save_word(self, index: int, input_word: str, output_word: str, example: str) -> bool:
        try:
            self.dataset['input'][index] = input_word
            if index >= len(self.dataset['output']):
                self.dataset['output'].extend([""] * (index - len(self.dataset['output']) + 1))
            self.dataset['output'][index] = output_word
            if 'examples' not in self.dataset:
                self.dataset['examples'] = [""] * len(self.dataset['input'])
            if index >= len(self.dataset['examples']):
                self.dataset['examples'].extend([""] * (index - len(self.dataset['examples']) + 1))
            self.dataset['examples'][index] = example
            if DatasetManager.save_dataset(self.dataset):
                self.stats = DatasetManager.get_statistics(self.dataset)
                return True
            return False
        except Exception as e:
            st.error(f"❌ Erreur lors de la sauvegarde: {str(e)}")
            return False

    def render(self):
        st.title("📝 Éditeur de Dataset")
        st.info(
            "Dans cette page, vous pouvez parcourir, éditer et enrichir le dictionnaire Antandroy-Français.\n\n"
            "Pour chaque mot, vous pouvez modifier la traduction, ajouter une phrase d'exemple et enregistrer votre voix pour créer un dataset vocal. "
            "Utilisez les boutons pour naviguer entre les mots et sauvegarder vos modifications."
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            current_index = st.session_state.get('current_index', 0)
            new_index = self.render_navigation(current_index)
            if new_index != current_index:
                st.session_state.current_index = new_index
                st.rerun()
            if current_index < self.stats['total']:
                new_index = self.render_edit_form(current_index)
                if new_index != current_index:
                    st.session_state.current_index = new_index
                    st.rerun()
        with col2:
            self.render_sidebar()

# Classe pour la collecte de textes
class TextCollector:
    def __init__(self):
        self.supported_dialects = ["Antandroy", "Sakalava", "Betsileo", "Merina", "Autres"]

    def render(self):
        st.title("📥 Collecte de Textes")
        st.info(
            "Utilisez cette page pour ajouter de nouveaux textes dans différents dialectes malgaches.\n\n"
            "Vous pouvez saisir du texte manuellement ou télécharger un fichier texte. "
            "Après traitement, les mots uniques seront extraits et sauvegardés pour enrichir le dataset."
        )
        dialect = st.selectbox("Sélectionnez le dialecte:", self.supported_dialects, key="dialect_selector")
        text_input = st.text_area("Entrez votre texte:", height=300, placeholder="Collez votre texte ici...", key="text_input_area")
        uploaded_file = st.file_uploader("Ou téléchargez un fichier texte:", type=['txt'], key="file_uploader")
        if uploaded_file:
            text_input = uploaded_file.getvalue().decode('utf-8')
        if st.button("🔄 Traiter le texte", key="process_button"):
            if text_input:
                try:
                    progress_container = st.empty()
                    result = process_text(text_input, progress_container)
                    output_dir = f"data/processed/{dialect.lower()}"
                    os.makedirs(output_dir, exist_ok=True)
                    output_file = f"{output_dir}/output_{int(time.time())}.json"
                    save_to_json(result, output_file)
                    progress_container.empty()
                    st.success(f"✅ Texte traité et sauvegardé dans {output_file}")
                    st.info(f"📊 Statistiques:\n- Mots uniques extraits: {len(result['input'])}\n- Phrases exemples trouvées: {len([x for x in result['examples'] if x])}")
                except Exception as e:
                    st.error(f"❌ Erreur lors du traitement: {str(e)}")
            else:
                st.warning("⚠️ Veuillez entrer du texte ou télécharger un fichier")

# Classe pour l'interface du chatbot
class ChatInterface:
    def __init__(self):
        self.model, self.tokenizer = ModelManager.load_model()
        if not st.session_state.get("history"):
            st.session_state.history = []

    def render_sidebar(self):
        st.sidebar.header("⚙️ Paramètres")
        max_length = st.sidebar.slider("Longueur maximale", 20, 200, DEFAULT_MAX_LENGTH)
        temperature = st.sidebar.slider("Température", 0.1, 1.0, DEFAULT_TEMPERATURE)
        st.sidebar.markdown("---")
        st.sidebar.subheader("ℹ️ À propos")
        st.sidebar.info("Ce chatbot utilise un modèle MT5 fine-tuné pour la langue Antandroy. Il a été entraîné sur un corpus spécialisé.")
        return max_length, temperature

    def render_chat_history(self):
        for msg in st.session_state.history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    def render_input_area(self, max_length: int, temperature: float):
        if prompt := st.chat_input("💬 Votre message:"):
            self.process_input(prompt, max_length, temperature)

    def process_input(self, user_input: str, max_length: int, temperature: float):
        st.session_state.history.append({"role": "user", "content": user_input})
        with st.chat_message("assistant"):
            with st.spinner("Génération de la réponse..."):
                response, response_time = ModelManager.generate_response(user_input, self.model, self.tokenizer, max_length, temperature)
            st.write(response)
            if response_time > 0:
                st.caption(f"⏱️ Temps de réponse: {response_time:.2f}s")
        st.session_state.history.append({"role": "assistant", "content": response})

    def render(self):
        st.title("🗣️ Chatbot Antandroy")
        st.info(
            "Bienvenue sur le Chatbot Antandroy !\n\n"
            "Posez vos questions ou saisissez des phrases en Antandroy ou en français. "
            "Le chatbot vous répondra grâce à un modèle MT5 spécialisé. "
            "Utilisez la barre latérale pour ajuster les paramètres de génération."
        )
        if not all([self.model, self.tokenizer]):
            st.error("❌ Impossible de continuer sans le modèle")
            st.stop()
        max_length, temperature = self.render_sidebar()
        self.render_chat_history()
        self.render_input_area(max_length, temperature)

def render_navigation() -> object:
    """
    Affiche le menu de navigation et retourne la classe de page sélectionnée.
    Options : 💬 Chatbot, 📝 Éditeur, 📥 Collecte.
    """
    menu_items = {
        "💬 Chatbot": ChatInterface,
        "📝 Éditeur": DatasetEditor,
        "📥 Collecte": TextCollector
    }
    nav_container = st.container()
    with nav_container:
        col1, col2, _ = st.columns([2, 8, 2])
        with col1:
            st.write("Navigation:")
        with col2:
            page = st.radio(label="Menu", options=list(menu_items.keys()), horizontal=True, key="nav_menu", label_visibility="collapsed")
    st.markdown("---")
    return menu_items[page]

def main():
    """Point d'entrée principal de l'application."""
    try:
        selected_page = render_navigation()
        page_instance = selected_page()
        page_instance.render()
    except Exception as e:
        st.error(f"❌ Erreur de l'application: {str(e)}")

if __name__ == "__main__":
    main()
