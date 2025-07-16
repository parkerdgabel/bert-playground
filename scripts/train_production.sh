#!/bin/bash
# Production-ready training script for Titanic ModernBERT

# Set environment variables for better performance
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=4

# Create output directory with timestamp
OUTPUT_DIR="./output/production_$(date +%Y%m%d_%H%M%S)"

# Production settings optimized for Titanic dataset (891 training samples)
# With augmentation, we have ~2673 samples
# Batch size 32 = ~84 steps per epoch
# 5 epochs = ~420 total steps (good for small dataset)

echo "Starting production training for Titanic dataset..."
echo "Dataset size: 891 samples (2673 with augmentation)"
echo "Batch size: 32"
echo "Epochs: 5"
echo "Expected steps: ~420"
echo "Output directory: $OUTPUT_DIR"

# Run training with production settings
uv run python train_titanic_v2.py \
    --train_path data/titanic/train.csv \
    --val_path data/titanic/val.csv \
    --model_name answerdotai/ModernBERT-base \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_epochs 5 \
    --output_dir "$OUTPUT_DIR" \
    --experiment_name titanic_production \
    --run_name "prod_run_$(date +%Y%m%d_%H%M%S)" \
    --log_level INFO \
    --do_train \
    2>&1 | tee "$OUTPUT_DIR/training.log"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Output directory: $OUTPUT_DIR"
    
    # Optional: Run predictions on test set
    echo "Running predictions on test set..."
    uv run python train_titanic_v2.py \
        --test_path data/titanic/test.csv \
        --checkpoint_path "$OUTPUT_DIR/best_model_accuracy" \
        --output_dir "$OUTPUT_DIR" \
        --do_predict \
        2>&1 | tee -a "$OUTPUT_DIR/training.log"
        
    # Display final metrics
    echo ""
    echo "=== Training Summary ==="
    if [ -f "$OUTPUT_DIR/training_history.json" ]; then
        echo "Final metrics:"
        uv run python -c "
import json
with open('$OUTPUT_DIR/training_history.json') as f:
    history = json.load(f)
    if history['train_loss']:
        print(f'  Final train loss: {history[\"train_loss\"][-1]:.4f}')
        print(f'  Final train accuracy: {history[\"train_accuracy\"][-1]:.4f}')
    if history['val_loss']:
        print(f'  Best val loss: {min(history[\"val_loss\"]):.4f}')
        print(f'  Best val accuracy: {max(history[\"val_accuracy\"]):.4f}')
"
    fi
    
    # Show submission file location if created
    if [ -f "$OUTPUT_DIR/submission.csv" ]; then
        echo ""
        echo "Kaggle submission file created: $OUTPUT_DIR/submission.csv"
    fi
else
    echo "Training failed! Check logs at $OUTPUT_DIR/training.log"
    exit 1
fi