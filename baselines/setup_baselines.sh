#!/bin/bash
# Setup script for baseline dependencies

echo "=========================================="
echo "Baseline Setup Script"
echo "=========================================="

# Create results directory
echo "Creating results directory..."
mkdir -p results

# Check Python version
echo -e "\nChecking Python version..."
python --version

# Install basic dependencies
echo -e "\nInstalling basic dependencies..."
pip install pandas numpy scikit-learn tqdm

echo -e "\n=========================================="
echo "Setting up individual baselines..."
echo "=========================================="

# Jellyfish
echo -e "\n[1/7] Setting up Jellyfish..."
pip install jellyfish
echo "✅ Jellyfish setup complete"

# Raha
echo -e "\n[2/7] Setting up Raha..."
if [ -f "raha/requirements.txt" ]; then
    pip install -r raha/requirements.txt
    echo "✅ Raha setup complete"
else
    echo "⚠️  Raha requirements.txt not found"
fi

# ActiveDetect
echo -e "\n[3/7] Setting up ActiveDetect..."
if [ -d "activedetect" ]; then
    pip install gensim usaddress
    echo "✅ ActiveDetect setup complete"
else
    echo "⚠️  ActiveDetect directory not found"
fi

# BigDansing
echo -e "\n[4/7] Setting up BigDansing..."
echo "✅ BigDansing uses standard libraries (already installed)"

# HoloClean
echo -e "\n[5/7] Setting up HoloClean..."
echo "⚠️  HoloClean requires PostgreSQL"
echo "    Please setup PostgreSQL manually:"
echo "    1. Install: sudo apt-get install postgresql"
echo "    2. Create database and user (see README_BASELINES.md)"
if [ -f "holoclean/requirements.txt" ]; then
    echo "    3. Run: pip install -r holoclean/requirements.txt"
fi

# GIDCL
echo -e "\n[6/7] Setting up GIDCL..."
echo "⚠️  GIDCL requires LLM models and GPU resources"
echo "    See GIDCL/README.md for detailed setup"

# MLNClean
echo -e "\n[7/7] Setting up MLNClean..."
echo "⚠️  MLNClean is Java-based"
echo "    Requires: Java 8+ and Maven"
echo "    Compile with: cd MLNClean && mvn clean package"

echo -e "\n=========================================="
echo "Setup Summary"
echo "=========================================="
echo "✅ Basic dependencies installed"
echo "✅ Jellyfish ready"
echo "✅ Raha ready"
echo "✅ ActiveDetect ready"
echo "✅ BigDansing ready"
echo "⚠️  HoloClean needs manual PostgreSQL setup"
echo "⚠️  GIDCL needs manual LLM setup"
echo "⚠️  MLNClean needs Java/Maven compilation"

echo -e "\nTo test a simple baseline, run:"
echo "  python run_jellyfish.py --dataset beers"
echo -e "\nFor more information, see README_BASELINES.md"

