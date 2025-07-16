#!/bin/bash
# Script to run MLX ModernBERT training as a background process

# Set default values
CONFIG="${1:-standard}"
EXPERIMENT_NAME="${2:-titanic_production}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./output/background_${CONFIG}_${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/training.log"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Display information
echo "=========================================="
echo "MLX ModernBERT Background Training"
echo "=========================================="
echo "Configuration: $CONFIG"
echo "Experiment: $EXPERIMENT_NAME"
echo "Output Directory: $OUTPUT_DIR"
echo "Log File: $LOG_FILE"
echo "=========================================="

# Function to run training
run_training() {
    # Export environment variables for optimal performance
    export TOKENIZERS_PARALLELISM=true
    export OMP_NUM_THREADS=4
    export LOGURU_COLORIZE=false  # Disable colors for log file
    
    # Run the training command
    uv run python mlx_bert_cli.py train \
        --train data/titanic/train.csv \
        --val data/titanic/val.csv \
        --output "$OUTPUT_DIR" \
        --experiment "$EXPERIMENT_NAME" \
        --config "configs/production.json" \
        2>&1 | tee "$LOG_FILE"
    
    # Check exit status
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "[$(date)] Training completed successfully!" >> "$LOG_FILE"
        
        # Run predictions if training succeeded
        echo "[$(date)] Running predictions on test set..." >> "$LOG_FILE"
        uv run python mlx_bert_cli.py predict \
            --test data/titanic/test.csv \
            --checkpoint "${OUTPUT_DIR}/best_model_accuracy" \
            --output "${OUTPUT_DIR}/submission.csv" \
            2>&1 | tee -a "$LOG_FILE"
            
        echo "[$(date)] All tasks completed! Results in: $OUTPUT_DIR" >> "$LOG_FILE"
    else
        echo "[$(date)] Training failed! Check log file: $LOG_FILE" >> "$LOG_FILE"
    fi
}

# Run training in background using nohup
echo "Starting training in background..."
nohup bash -c "$(declare -f run_training); run_training" > /dev/null 2>&1 &

# Get the process ID
PID=$!
echo "Training started with PID: $PID"

# Save process info
echo "$PID" > "${OUTPUT_DIR}/training.pid"
echo "Process ID saved to: ${OUTPUT_DIR}/training.pid"

# Create monitoring script
cat > "${OUTPUT_DIR}/monitor.sh" << 'EOF'
#!/bin/bash
# Monitor training progress

PID=$(cat training.pid 2>/dev/null)
LOG_FILE="training.log"

if [ -z "$PID" ]; then
    echo "No PID found"
    exit 1
fi

echo "Monitoring training process (PID: $PID)"
echo "Press Ctrl+C to stop monitoring (training will continue)"
echo "=========================================="

# Check if process is running
if ps -p $PID > /dev/null; then
    echo "Training is running..."
    echo ""
    # Follow the log file
    tail -f "$LOG_FILE"
else
    echo "Training process is not running"
    echo "Last 20 lines of log:"
    tail -20 "$LOG_FILE"
fi
EOF

chmod +x "${OUTPUT_DIR}/monitor.sh"

# Create stop script
cat > "${OUTPUT_DIR}/stop.sh" << 'EOF'
#!/bin/bash
# Stop training process

PID=$(cat training.pid 2>/dev/null)

if [ -z "$PID" ]; then
    echo "No PID found"
    exit 1
fi

if ps -p $PID > /dev/null; then
    echo "Stopping training process (PID: $PID)..."
    kill $PID
    echo "Process stopped"
else
    echo "Process is not running"
fi
EOF

chmod +x "${OUTPUT_DIR}/stop.sh"

echo ""
echo "Training is running in background!"
echo ""
echo "To monitor progress:"
echo "  cd $OUTPUT_DIR && ./monitor.sh"
echo ""
echo "To check status:"
echo "  ps -p $PID"
echo ""
echo "To view logs:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To stop training:"
echo "  cd $OUTPUT_DIR && ./stop.sh"
echo ""
echo "Results will be saved to: $OUTPUT_DIR"