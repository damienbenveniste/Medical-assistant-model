import argparse
from data import DataConnector
from model import Model
from training import SupervisedTrainer, EncDecTrainer
from data_processing import DataProcessor
from dotenv import load_dotenv

load_dotenv()


def run_decoder_only():
    """
    Run decoder-only model training pipeline.
    
    Loads data, splits into train/val/test, loads decoder model with tokenizer,
    and trains using SupervisedTrainer with LoRA adapters.
    """
    data_connector = DataConnector()
    data = data_connector.load()
    data = data_connector.split(data)

    tokenizer, model = Model.load()

    trainer = SupervisedTrainer(model, tokenizer)
    trainer.train(data)

def run_encoder_decoder():
    """
    Run encoder-decoder model training pipeline.
    
    Loads data, splits into train/val/test, loads seq2seq model with tokenizer,
    processes data for seq2seq format, and trains using EncDecTrainer.
    """
    data_connector = DataConnector()
    data = data_connector.load()
    data = data_connector.split(data)

    tokenizer, model = Model.load_seq2seq()
    data_processor = DataProcessor(tokenizer)
    tokenized_data = data_processor.transform(data)

    trainer = EncDecTrainer(model, tokenizer)
    trainer.train(tokenized_data)


def main():
    """
    Main entry point for training medical assistant models.
    
    Parses command line arguments to determine model type and runs
    the appropriate training pipeline (decoder-only or encoder-decoder).
    
    Command line usage:
        python main.py --model-type decoder        # Default: decoder-only training
        python main.py --model-type encoder-decoder # Seq2seq training
    """
    parser = argparse.ArgumentParser(description="Train medical assistant model")
    parser.add_argument(
        "--model-type", 
        choices=["decoder", "encoder-decoder"], 
        default="decoder",
        help="Type of model to train (default: decoder)"
    )
    
    args = parser.parse_args()
    
    if args.model_type == "decoder":
        print("Running decoder-only training...")
        run_decoder_only()
    else:
        print("Running encoder-decoder training...")
        run_encoder_decoder()


if __name__ == '__main__':
    main()

