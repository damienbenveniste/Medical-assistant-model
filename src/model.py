from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM


class Model:

    @ staticmethod
    def load(model_id: str = "Qwen/Qwen3-0.6B-Base"):
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        tokenizer.padding_side = "left"
        return tokenizer, model
    
    def load_seq2seq(model_id: str = "google/flan-t5-small"):
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        return tokenizer, model