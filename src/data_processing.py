from typing import Any
from datasets import Dataset
from transformers import PreTrainedTokenizerBase


class DataProcessor:
    """Processes datasets for sequence-to-sequence training by tokenizing prompts and completions."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """
        Initialize the data processor with a tokenizer.
        
        Args:
            tokenizer: Pre-trained tokenizer for encoding text
        """
        self.tokenizer = tokenizer

    def _tokenize(self, samples: dict[str, str]) -> dict[str, Any]:
        """
        Tokenize prompt and completion pairs for seq2seq training.
        
        Args:
            samples: Dictionary containing 'prompt' and 'completion' keys
            
        Returns:
            Dictionary with tokenized inputs and labels for model training
            
        Note:
            Uses max_length=256 with padding and truncation
        """
        model_inputs = self.tokenizer(
            samples['prompt'], 
            padding='max_length',
            truncation=True,
            max_length=256
        )
        
        labels = self.tokenizer(
            text_target=samples['completion'], 
            padding='max_length',
            truncation=True,
            max_length=256,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def _filter(self, sample: dict[str, str | None]) -> bool:
        """
        Filter out samples with empty prompts or completions.
        
        Args:
            sample: Dictionary containing prompt and completion
            
        Returns:
            True if both prompt and completion are non-empty, False otherwise
        """
        return sample["prompt"] and sample["completion"]

    def transform(self, data: Dataset) -> Dataset:
        """
        Transform dataset by filtering and tokenizing samples.
        
        Args:
            data: Input dataset with prompt/completion pairs
            
        Returns:
            Processed dataset ready for seq2seq training
            
        Note:
            Applies filtering first, then tokenization
        """
        return data.filter(self._filter).map(self._tokenize, batched=False)


    
