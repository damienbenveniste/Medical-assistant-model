from datasets import load_dataset, DatasetDict
from pathlib import Path


data_root = Path(__file__).resolve().parent.parent
FILE_PATH  = data_root / "data" / "medical_data.csv"


class DataConnector:
    """Handles loading and preprocessing of medical Q&A datasets."""

    def load(self, file_name: str = str(FILE_PATH)) -> DatasetDict:
        """
        Load medical Q&A data from CSV file and preprocess it.
        
        Args:
            file_name: Path to CSV file containing question/answer pairs
            
        Returns:
            DatasetDict with preprocessed data (columns renamed to prompt/completion)
            
        Note:
            Filters out rows where question or answer is empty/None
        """
        data = load_dataset("csv", data_files=file_name)
        data = data.filter(lambda x: x["question"] and x["answer"])
        data = data.rename_column('question', 'prompt').rename_column('answer', 'completion')
        return data
    
    def split(self, dataset: DatasetDict, test: float = 0.05, val: float = 0.01, seed=42) -> DatasetDict:
        """
        Split dataset into train/validation/test splits.
        
        Args:
            dataset: Input DatasetDict to split
            test: Fraction for test set (default: 0.05 = 5%)
            val: Fraction for validation set (default: 0.01 = 1%) 
            seed: Random seed for reproducible splits
            
        Returns:
            DatasetDict with train/val/test splits
            
        Example:
            With default values: 94% train, 1% val, 5% test
        """
        test_val_rate = test + val
        test_rate = test / test_val_rate

        train_temp = dataset['train'].train_test_split(test_size=test_val_rate, seed=seed)
        test_val = train_temp["test"].train_test_split(test_size=test_rate, seed=seed)

        return DatasetDict({
            "train": train_temp["train"],
            "test": test_val["test"],
            "val": test_val["train"]
        })
    

