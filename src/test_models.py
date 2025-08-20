"""
Test script for evaluating trained medical assistant models on sample questions.
Tests both QWen (decoder-only) and T5-FLAN (encoder-decoder) models using transformers pipeline.
"""

import pandas as pd
from transformers import pipeline
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

QWEN_BASE = "Qwen/Qwen3-0.6B-Base"
ADAPTER_REPO = "damienbenveniste/medical_assistant"   
SUBFOLDER = "medical"   
FLAN_REPO = "damienbenveniste/medical_assistant_seq2seq"


def load_sample_questions(csv_path: str, n_samples: int = 3) -> list[dict]:
    """
    Load sample questions from the medical dataset.
    
    Args:
        csv_path: Path to the medical_data.csv file
        n_samples: Number of sample questions to load
        
    Returns:
        List of dictionaries with 'question' and 'expected_answer' keys
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['question', 'answer'])
    samples = df.sample(n=n_samples, random_state=42)
    
    return [
        {
            'question': row['question'], 
            'expected_answer': row['answer']
        } 
        for _, row in samples.iterrows()
    ]


def test_qwen_model(questions: list[str]) -> list[str]:
    """
    Test the QWen decoder-only model on given questions.
    
    Args:
        questions: List of medical questions
        
    Returns:
        List of generated answers
    """
          
    tokenizer = AutoTokenizer.from_pretrained(QWEN_BASE, use_fast=True)
    tokenizer.padding_side = "left" 
    base_model = AutoModelForCausalLM.from_pretrained(QWEN_BASE)
    model = PeftModel.from_pretrained(base_model, ADAPTER_REPO, subfolder=SUBFOLDER)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )

    answers = []
    for question in questions:
        response = generator(question, max_new_tokens=200)[0]['generated_text']
        # Remove the original question from the response to get only generated text
        generated_answer = response[len(question):].strip()
        answers.append(generated_answer)
    
    return answers


def test_flan_model(questions: list[str]) -> list[str]:
    """
    Test the T5-FLAN encoder-decoder model on given questions.
    
    Args:
        questions: List of medical questions
        
    Returns:
        List of generated answers
    """
    
    generator = pipeline(
        "text2text-generation",
        model=FLAN_REPO,
        tokenizer=FLAN_REPO,
    )
    
    answers = []
    for question in questions:
        response = generator(question, max_new_tokens=200)[0]['generated_text']
        answers.append(response)
    
    return answers


def save_results_to_file(samples: list[dict], qwen_answers: list[str], flan_answers: list[str]):
    """Save the comparison results to a text file."""
    output_file = f"model_test_results.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("MEDICAL ASSISTANT MODEL COMPARISON\n")
        f.write("="*100 + "\n")
        f.write(f"Models tested: QWen vs T5-FLAN\n\n")
        
        for i, (sample, qwen_ans, flan_ans) in enumerate(zip(samples, qwen_answers, flan_answers), 1):
            f.write(f"{'='*20} QUESTION {i} {'='*20}\n")
            f.write(f"QUESTION: {sample['question']}\n\n")
            
            f.write("EXPECTED ANSWER:\n")
            f.write(f"{sample['expected_answer']}\n\n")
            
            f.write("QWEN MODEL ANSWER:\n")
            f.write(f"{qwen_ans}\n\n")
            
            f.write("T5-FLAN MODEL ANSWER:\n")
            f.write(f"{flan_ans}\n\n")
            
            f.write("-" * 80 + "\n\n")


def main():
    """Main function to run the model testing."""
    data_path = Path(__file__).parent.parent / "data" / "medical_data.csv"
    samples = load_sample_questions(str(data_path), n_samples=3)
    questions = [sample['question'] for sample in samples]
    qwen_answers = test_qwen_model(questions)    
    flan_answers = test_flan_model(questions)
    save_results_to_file(samples, qwen_answers, flan_answers)
                  
if __name__ == "__main__":
    main()