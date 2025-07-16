#!/usr/bin/env python3
import pandas as pd
from data.text_templates import TitanicTextTemplates
from pathlib import Path

def demo_text_conversion():
    print("=== Titanic BERT Demo ===\n")
    
    # Load sample data
    data_path = Path("kaggle/titanic/data/train.csv")
    if not data_path.exists():
        print("Please run 'python kaggle/titanic/download_data.py' first!")
        return
    
    df = pd.read_csv(data_path)
    
    # Initialize text converter
    converter = TitanicTextTemplates()
    
    print("Sample passenger data to text conversions:\n")
    
    # Convert first few passengers to text
    for idx in range(min(3, len(df))):
        row = df.iloc[idx]
        print(f"Passenger {idx + 1}:")
        print(f"  Original data: {row['Name']}, {row['Sex']}, Age: {row['Age']}, Class: {row['Pclass']}")
        print(f"  Survived: {'Yes' if row['Survived'] else 'No'}")
        
        # Generate multiple text representations
        print("  Text representations:")
        for template_idx in range(min(2, len(converter.templates))):
            text = converter.row_to_text(row.to_dict(), template_idx=template_idx)
            print(f"    - {text}")
        print()
    
    print("\n=== MLX ModernBERT Architecture ===")
    print("The model architecture includes:")
    print("- ModernBERT encoder (12 layers, 768 hidden size)")
    print("- Binary classification head")
    print("- Optimized for Apple Silicon with MLX")
    print("\nTo train the model, run:")
    print("  uv run python train_titanic.py --do_train --do_predict")
    print("\nTo generate predictions only:")
    print("  uv run python train_titanic.py --do_predict --checkpoint_path ./output/best_model")


if __name__ == "__main__":
    demo_text_conversion()