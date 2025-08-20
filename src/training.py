from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
import evaluate
import numpy as np
from huggingface_hub import login
from transformers import (
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizer
)
import os
import evaluate
import torch
from typing import Any
from datasets import DatasetDict
from torch import Tensor



class GenSFTTrainer(SFTTrainer):
    """
    Custom SFT trainer that generates tokens during prediction for evaluation metrics.
    Extends the base SFTTrainer to include text generation capabilities.
    """
    
    def prediction_step(
        self, 
        model: PreTrainedModel, 
        inputs: dict[str, Any], 
        prediction_loss_only: bool, 
        ignore_keys: list | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Perform prediction step with text generation for evaluation.
        
        Args:
            model: The model to use for prediction
            inputs: Input tensors (input_ids, attention_mask, etc.)
            prediction_loss_only: Whether to only compute loss (ignored, always generates)
            ignore_keys: Keys to ignore in outputs
            
        Returns:
            Tuple of (loss, generated_tokens, labels)
        """
        out = super().prediction_step(model, inputs, False, ignore_keys)
        loss, _, labels = (out if isinstance(out, tuple)
                           else (out.loss, out.predictions, out.label_ids))
        with torch.no_grad():
            gen_tokens = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],  
                max_new_tokens=128
            )
        return (loss, gen_tokens, labels)


class SupervisedTrainer:
    """
    Trainer for supervised fine-tuning of decoder-only models using LoRA adapters.
    Handles model training with BERTScore evaluation and Hugging Face Hub integration.
    """

    def __init__(
            self, 
            model: PreTrainedModel, 
            tokenizer: PreTrainedTokenizer, 
            num_epoch: int = 3, 
            batch_size: int = 8, 
            output_dir: str = 'medical_assistant',
        ) -> None:
        """
        Initialize the supervised trainer.
        
        Args:
            model: Pre-trained model to fine-tune
            tokenizer: Tokenizer for text processing
            num_epoch: Number of training epochs (default: 3)
            batch_size: Training batch size (default: 8)
            output_dir: Output directory for model checkpoints (default: 'medical_assistant')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.output_dir = output_dir
        
        self.args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=num_epoch,
            per_device_train_batch_size=batch_size,
            max_length=256,
            dataloader_pin_memory=False,
            eval_strategy="epoch",            
            save_strategy="epoch",                 
            logging_strategy="epoch", 
            load_best_model_at_end=True,          
            metric_for_best_model="bertscore_f1",    
            greater_is_better=True,   
            push_to_hub=True,
            report_to="none", 
            bf16=False,
            completion_only_loss=True,
            group_by_length=True,
        )

        self.lora_config =  LoraConfig(
            r=64,
            task_type="CAUSAL_LM", 
            target_modules='all-linear'
        )
        self.bertscore = evaluate.load("bertscore")

    def add_adapter(self) -> PreTrainedModel:
        """
        Add LoRA adapter to the model for parameter-efficient fine-tuning.
        
        Returns:
            Model with LoRA adapter attached
            
        Note:
            Uses LoRA rank=64 targeting all linear layers
        """
        peft_model = get_peft_model(
            self.model, 
            self.lora_config, 
            adapter_name='medical'
        )
        return peft_model

    def train(self, dataset: DatasetDict) -> None:
        """
        Train the model using supervised fine-tuning.
        
        Args:
            dataset: DatasetDict containing train/val splits
            
        Note:
            - Adds LoRA adapter before training
            - Logs in to Hugging Face Hub if HF_TOKEN is set
            - Pushes final model to Hub after training
        """

        peft_model = self.add_adapter()

        token = os.environ.get('HF_TOKEN')
        if token:
            login(token=token)

        trainer = GenSFTTrainer(
            peft_model,
            args=self.args,
            processing_class=self.tokenizer,
            train_dataset=dataset['train'],
            eval_dataset=dataset['val'],
            compute_metrics=self.compute_metrics,
            
        )

        trainer.train()
        trainer.push_to_hub()

    def compute_metrics(self, eval_pred: tuple[Tensor | np.ndarray, Tensor | np.ndarray]) -> dict[str, float]:
        """
        Compute BERTScore F1 metric for evaluation.
        
        Args:
            eval_pred: Tuple of (predictions, labels) tensors or arrays
            
        Returns:
            Dictionary containing 'bertscore_f1' metric
            
        Note:
            Uses DeBERTa-XLarge-MNLI model for BERTScore computation
        """
        preds, labels = eval_pred

        if isinstance(preds, torch.Tensor): preds = preds.cpu().numpy()
        if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()

        if isinstance(preds, (list, tuple)):
            preds = preds[0] if isinstance(preds, tuple) else np.array(preds)

        pad_id = self.tokenizer.pad_token_id
        preds = np.where(preds < 0, pad_id, preds)
        labels = np.where(labels < 0, pad_id, labels)

        decoded_preds  = self.tokenizer.batch_decode(preds,  skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        score = self.bertscore.compute(
            predictions=decoded_preds, 
            references=decoded_labels, 
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli"
        )
        return {"bertscore_f1": float(np.mean(score["f1"]))}


class EncDecTrainer:
    """
    Trainer for encoder-decoder models (T5, BART, etc.) for sequence-to-sequence tasks.
    Handles seq2seq training with BERTScore evaluation and Hugging Face Hub integration.
    """

    def __init__(
            self, 
            model: PreTrainedModel, 
            tokenizer: PreTrainedTokenizer, 
            num_epoch: int = 5, 
            batch_size: int = 8, 
            output_dir: str = 'medical_assistant_seq2seq',
        ) -> None:
        """
        Initialize the encoder-decoder trainer.
        
        Args:
            model: Pre-trained encoder-decoder model
            tokenizer: Tokenizer for text processing
            num_epoch: Number of training epochs (default: 5)
            batch_size: Training batch size (default: 8)
            output_dir: Output directory for model checkpoints (default: 'medical_assistant_seq2seq')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.output_dir = output_dir
        
        self.args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epoch,
            per_device_train_batch_size=batch_size,
            dataloader_pin_memory=False,
            eval_strategy="epoch",            
            save_strategy="epoch",                 
            logging_strategy="epoch", 
            load_best_model_at_end=True,          
            metric_for_best_model="bertscore_f1",    
            greater_is_better=True,   
            push_to_hub=True,
            report_to="none", 
            bf16=False,
            group_by_length=True,
            predict_with_generate=True,
            generation_max_length=256,
        )
        self.bertscore = evaluate.load("bertscore")
        self.data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    def train(self, dataset: DatasetDict) -> None:
        """
        Train the encoder-decoder model.
        
        Args:
            dataset: DatasetDict containing train/val splits
            
        Note:
            - Uses DataCollatorForSeq2Seq for proper padding
            - Logs in to Hugging Face Hub if HF_TOKEN is set
            - Pushes final model to Hub after training
        """

        token = os.environ.get('HF_TOKEN')
        if token:
            login(token=token)

        trainer = Seq2SeqTrainer(
            self.model,
            args=self.args,
            processing_class=self.tokenizer,
            train_dataset=dataset['train'],
            eval_dataset=dataset['val'],
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics, 
        )

        trainer.train()
        trainer.push_to_hub()

    def compute_metrics(self, eval_pred: tuple[Tensor | np.ndarray, Tensor | np.ndarray]) -> dict[str, float]:
        """
        Compute BERTScore F1 metric for evaluation.
        
        Args:
            eval_pred: Tuple of (predictions, labels) tensors or arrays
            
        Returns:
            Dictionary containing 'bertscore_f1' metric
            
        Note:
            Uses DeBERTa-XLarge-MNLI model for BERTScore computation
        """
        preds, labels = eval_pred

        if isinstance(preds, torch.Tensor): preds = preds.cpu().numpy()
        if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()

        if isinstance(preds, (list, tuple)):
            preds = preds[0] if isinstance(preds, tuple) else np.array(preds)

        pad_id = self.tokenizer.pad_token_id
        preds = np.where(preds < 0, pad_id, preds)
        labels = np.where(labels < 0, pad_id, labels)

        decoded_preds  = self.tokenizer.batch_decode(preds,  skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        score = self.bertscore.compute(
            predictions=decoded_preds, 
            references=decoded_labels, 
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli"
        )
        return {"bertscore_f1": float(np.mean(score["f1"]))}