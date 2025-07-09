import pytest
import sys
import os
from pathlib import Path
import json
import torch
from unittest.mock import patch, MagicMock, mock_open
from transformers import MT5ForConditionalGeneration, AutoTokenizer
from scripts.6_streamlit_chatbot import ModelManager, DatasetManager, TextCollector

# filepath: scripts/test_6_streamlit_chatbot.py


# Add parent directory to path to import main script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock streamlit
class MockStreamlit:
    def __init__(self):
        self.sidebar = MagicMock()
        self.session_state = {}
        
    def cache_resource(self, func):
        return func
        
    def spinner(self, text):
        class SpinnerContext:
            def __enter__(self): pass
            def __exit__(self, *args): pass
        return SpinnerContext()

@pytest.fixture
def mock_st():
    with patch('scripts.6_streamlit_chatbot.st', MockStreamlit()) as mock:
        yield mock

@pytest.fixture
def sample_dataset():
    return {
        "input": ["mot1", "mot2", "mot3"],
        "output": ["word1", "word2", ""],
        "examples": ["example1", "example2", ""]
    }

@pytest.fixture
def mock_model():
    model = MagicMock(spec=MT5ForConditionalGeneration)
    model.generate.return_value = torch.tensor([[1, 2, 3]])
    return model

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock(spec=AutoTokenizer)
    tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
    tokenizer.decode.return_value = "Réponse test"
    return tokenizer

def test_model_manager_load_model(mock_st, mock_model, mock_tokenizer):
    with patch('transformers.MT5ForConditionalGeneration.from_pretrained') as mock_model_load:
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer_load:
            mock_model_load.return_value = mock_model
            mock_tokenizer_load.return_value = mock_tokenizer
            
            model, tokenizer = ModelManager.load_model()
            
            assert model == mock_model
            assert tokenizer == mock_tokenizer
            mock_model_load.assert_called_once()
            mock_tokenizer_load.assert_called_once()

def test_model_manager_generate_response(mock_st, mock_model, mock_tokenizer):
    response, time_taken = ModelManager.generate_response(
        "Test input",
        mock_model,
        mock_tokenizer,
        max_length=128,
        temperature=0.7
    )
    
    assert isinstance(response, str)
    assert isinstance(time_taken, float)
    assert response == "Réponse test"
    mock_model.generate.assert_called_once()

def test_dataset_manager_load_dataset(mock_st, sample_dataset):
    with patch('builtins.open', mock_open(read_data=json.dumps(sample_dataset))):
        dataset = DatasetManager.load_dataset()
        assert dataset == sample_dataset

def test_dataset_manager_save_dataset(mock_st, sample_dataset):
    mock_file = mock_open()
    with patch('builtins.open', mock_file):
        result = DatasetManager.save_dataset(sample_dataset)
        assert result is True
        mock_file.assert_called_once()

def test_dataset_manager_get_statistics(mock_st, sample_dataset):
    stats = DatasetManager.get_statistics(sample_dataset)
    assert stats["total"] == 3
    assert stats["translated"] == 2
    assert stats["progress"] == (2/3) * 100

def test_text_collector_initialization(mock_st):
    collector = TextCollector()
    assert "Antandroy" in collector.supported_dialects
    assert len(collector.supported_dialects) == 5

@pytest.mark.asyncio
async def test_process_text_input(mock_st):
    with patch('scripts.6_streamlit_chatbot.process_text') as mock_process:
        mock_process.return_value = {
            "input": ["test"],
            "output": [""],
            "examples": [""]
        }
        
        collector = TextCollector()
        with patch('os.makedirs'):
            with patch('scripts.6_streamlit_chatbot.save_to_json') as mock_save:
                # Simulate text processing
                result = mock_process("Test input")
                assert "input" in result
                mock_process.assert_called_once()

def test_error_handling(mock_st):
    with pytest.raises(Exception):
        with patch('transformers.MT5ForConditionalGeneration.from_pretrained') as mock_load:
            mock_load.side_effect = Exception("Test error")
            ModelManager.load_model()

if __name__ == '__main__':
    pytest.main(['-v'])