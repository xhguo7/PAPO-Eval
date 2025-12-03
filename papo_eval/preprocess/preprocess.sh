#!/bin/bash

# =============================================================================
# Download PAPO evaluation datasets from HuggingFace
# =============================================================================

# Configuration variables - MODIFY THESE AS NEEDED
REPO_ID="PAPO-Galaxy/PAPO_eval"
OUTPUT_DIR="./data"  # If you change this to other places, may need to manually move back to here when running evaluation
AUTO_UNZIP=true  # Set to true to auto-unzip images and remove ZIP files

# Available dataset splits (uncomment the one you want to download)
SPLIT_NAME="hiyouga_geometry3k"
# SPLIT_NAME="AI4Math_MathVerse"
# SPLIT_NAME="AI4Math_MathVista"
# SPLIT_NAME="We_Math"
# SPLIT_NAME="AI4Math_MathVerse_vision_dependent"
# SPLIT_NAME="BUAADreamer_clevr_count_70k"
# SPLIT_NAME="lscpku_LogicVista"
# SPLIT_NAME="MMMU_MMMU_Pro"
# SPLIT_NAME="PAPO_MMK12"


# =============================================================================
# Script execution - DO NOT MODIFY BELOW THIS LINE
# =============================================================================

echo "🚀 Starting dataset download..."
echo "Repository: $REPO_ID"
echo "Split: $SPLIT_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Auto unzip: $AUTO_UNZIP"
echo "----------------------------------------"

# Check if Python script exists
if [ ! -f "./papo_eval/preprocess/prepare_data.py" ]; then
    echo "❌ Error: prepare_data.py not found in current directory"
    exit 1
fi

# Check if required packages are installed
python3 -c "import datasets, huggingface_hub" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Required packages not installed"
    echo "Please run: pip install datasets huggingface_hub"
    exit 1
fi

# Build the command
CMD="python ./papo_eval/preprocess/prepare_data.py --repo_id \"$REPO_ID\" --split_name \"$SPLIT_NAME\" --output_dir \"$OUTPUT_DIR\""

# Add auto_unzip flag if true
if [ "$AUTO_UNZIP" = true ]; then
    CMD="$CMD --auto_unzip"
fi

# Execute the command
echo "Executing: $CMD"
echo "----------------------------------------"
eval $CMD

# Check if command was successful
if [ $? -eq 0 ]; then
    echo "----------------------------------------"
    echo "✅ Dataset download completed successfully!"
    echo "📁 Check your data in: $OUTPUT_DIR/$SPLIT_NAME/"
else
    echo "----------------------------------------"
    echo "❌ Dataset download failed!"
    exit 1
fi

# =============================================================================
# Utility functions
# =============================================================================

# Function to download all datasets
download_all_datasets() {
    echo "🔄 Downloading all datasets..."
    
    SPLITS=("AI4Math_MathVerse" "AI4Math_MathVerse_vision_dependent" "AI4Math_MathVista" 
            "BUAADreamer_clevr_count_70k" "hiyouga_geometry3k" "lscpku_LogicVista" 
            "MMMU_MMMU_Pro" "We_Math")
    
    for split in "${SPLITS[@]}"; do
        echo ""
        echo "📦 Processing: $split"
        echo "----------------------------------------"
        
        CMD="python ./papo_eval/preprocess/prepare_data.py --repo_id \"$REPO_ID\" --split_name \"$split\" --output_dir \"$OUTPUT_DIR\""
        
        if [ "$AUTO_UNZIP" = true ]; then
            CMD="$CMD --auto_unzip"
        fi
        
        eval $CMD
        
        if [ $? -ne 0 ]; then
            echo "❌ Failed to download $split"
        else
            echo "✅ Successfully downloaded $split"
        fi
    done
    
    echo ""
    echo "🎉 All datasets processing completed!"
}

# Function to list available splits
list_splits() {
    echo "📋 Available dataset splits:"
    python ./papo_eval/preprocess/prepare_data.py --list_splits
}

# =============================================================================
# Usage Instruction for PAPO Evaluation
# =============================================================================

# To download all datasets:
# download_all_datasets

# To list available splits:
# list_splits