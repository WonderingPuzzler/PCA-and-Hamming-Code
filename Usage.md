# PCA & Hamming Code Implementation Setup Guide

This project contains pure Python implementations of Principal Component Analysis (PCA) and Hamming error-correcting codes, along with an image classification system using eigenfaces.

## Prerequisites

- **Python 3.9 or higher** (I used Python 3.13)
- **pip** (Python package installer)
- **Git** (optional, for cloning the repository)

## Quick Start

### Option 1: Using Git (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/WonderingPuzzler/PCA-and-Hamming_Code.git
cd PCA-and-Hamming-Code

# 2. Create a virtual environment (recommended)
python -m venv .venv

# 3. Activate the virtual environment
# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

# 4. Install required packages
pip install -r requirements.txt

# 5. Run the programs (see usage section below)
```

### Option 2: Manual Setup

```bash
# 1. Download the project files to a folder named PCA-and-Hamming-Code
# 2. Navigate to the project directory
cd PCA-and=Hamming-Code

# 3. Create a virtual environment (recommended)
python -m venv .venv

# 4. Activate the virtual environment
# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

# 5. Install required packages
pip install -r requirements.txt
```

## Usage

### 1. Vector Matrix Operations Demo

```bash
python vector_matrix_operations.py
```
**What it does:** Demonstrates all basic linear algebra operations including vector operations, matrix operations, and determinant calculations.

### 2. PCA Algorithm Demo

```bash
python pca_algorithm.py
```
**What it does:** Shows step-by-step PCA analysis on sample 3D data, including mean calculation, data centering, covariance matrix computation, and eigenvalue decomposition.

### 3. Hamming Code Demo

```bash
python hamming_code.py
```
**What it does:** Demonstrates (7,4) Hamming error-correcting code encoding, decoding, and error correction capabilities.

### 4. PCA Image Classification Demo

```bash
python pca_image_classification.py
```
**What it does:** Performs face recognition using PCA (eigenfaces) and k-nearest neighbors classification on the LFW dataset. (Warning, because of the size of matrix multiplication we're doing, this program takes quite awhile)

## Troubleshooting

1. **"ModuleNotFoundError"**
   ```bash
   # Make sure virtual environment is activated and dependencies are installed
   pip install -r requirements.txt
   ```

2. **"Python not found"**
   ```bash
   # Try using py instead of python (Windows)
   py -m venv .venv
   py -m pip install -r requirements.txt
   ```

3. **Permission errors**
   ```bash
   # Run command prompt as administrator or use:
   python -m pip install --user -r requirements.txt
   ```
