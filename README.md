# Function Co-Change Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange)](https://xgboost.readthedocs.io/)

A machine learning framework for predicting function co-change relationships in software projects using static code analysis and historical commit data.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Phase 0: Repository Setup (Optional)](#phase-0-repository-setup-optional)
  - [Phase 1: Static Code Analysis](#phase-1-static-code-analysis)
  - [Phase 2: Function Feature Extraction](#phase-2-function-feature-extraction)
  - [Phase 3: Label Generation](#phase-3-label-generation)
  - [Phase 4: Pairwise Feature Construction](#phase-4-pairwise-feature-construction)
  - [Phase 5: Model Training](#phase-5-model-training)
  - [Phase 6: Inference & Prediction](#phase-6-inference--prediction)
- [Multi-Project Joint Training](#multi-project-joint-training)
- [License](#license)

## Overview

This project implements a novel approach to predict which functions are likely to be modified together (co-changed) during software evolution. By combining static code analysis with machine learning techniques, the system can:

- **Predict coupling relationships** between functions without relying solely on runtime behavior
- **Support both hybrid and static modes**: Use historical commit data when available, or rely purely on static analysis for new projects
- **Achieve cross-project generalization**: Train on multiple projects and apply to unseen projects with strong performance
- **Generate actionable insights**: Output co-change prediction graphs that can guide refactoring and impact analysis

### Key Applications

- **Impact Analysis**: Identify functions that may need modification when changing a specific function
- **Refactoring Guidance**: Discover hidden coupling relationships in legacy code
- **Code Review Assistance**: Suggest related files/functions to review together
- **Technical Debt Detection**: Reveal architectural inconsistencies through unexpected coupling patterns

## Features

- **Multi-language Support**: Analyze Python, Java, JavaScript, C/C++ code using Tree-sitter AST parsing
- **Call Graph Construction**: Build comprehensive function call graphs with detailed metadata
- **Rich Feature Engineering**: Extract structural, syntactic, and semantic features from function pairs
- **Flexible Training Modes**:
  - **Hybrid Mode**: Leverage real commit history for accurate label generation
  - **Static Mode**: Generate labels based on code structure alone (no git history required)
- **XGBoost-based Prediction**: High-performance gradient boosting classifier with excellent interpretability
- **Cross-Project Transfer Learning**: Multi-project joint training significantly improves generalization
- **Visualization Support**: Export prediction results as GraphML for visualization in Gephi or NetworkX

## Project Structure

```
ECE4010/
├── data/                                    # Core data processing scripts
│   ├── data_cleaning.py                     # Phase 0: Clone repos & extract commits
│   ├── static_code_analysis.py              # Phase 1: Static analysis & call graph construction
│   ├── function_feature_extraction.py       # Phase 2: Function-level feature extraction
│   ├── label_generation.py                  # Phase 3: Training label generation
│   ├── pairwise_features.py                 # Phase 4: Pairwise feature construction
│   └── code_embedder.py                     # CodeBERT embedder (optional)
├── train/
│   ├── train_xgboost.py                     # Phase 5: Single-project XGBoost training
│   └── train_joint.py                       # Multi-project joint training
├── verification/
│   ├── predict_cochange_graph.py            # Phase 6: Co-change prediction
│   └── evaluate_cross_project.py            # Cross-project evaluation
├── model/                                   # Shared models for cross-project use
│   ├── xgboost.pkl                          # Default trained model
│   ├── xgboost_joint.pkl                    # Joint-trained model
│   ├── scaler_joint.pkl                     # Joint feature scaler
│   └── features/
│       └── scaler.pkl                       # Default feature scaler
├── projects/                                # Project data storage (gitignored)
│   └── <project_name>/                      # Individual project directory
│       ├── source_code/                     # Cloned source code (Phase 0)
│       ├── commits.json                     # Commit history (Phase 0)
│       ├── static_analysis/                 # Static analysis results (Phase 1)
│       │   ├── call_graph.pkl
│       │   └── function_metadata.json
│       ├── features/                        # Function features (Phase 2 & 4)
│       │   ├── function_features.npy
│       │   ├── function_ids.json
│       │   ├── X_train.npy, y_train.npy
│       │   ├── X_val.npy, y_val.npy
│       │   ├── X_test.npy, y_test.npy
│       │   └── scaler.pkl
│       ├── labels/                          # Training labels (Phase 3)
│       │   ├── training_labels_v2.json
│       │   ├── positive_pairs.json
│       │   ├── negative_pairs.json
│       │   └── commit_to_functions.json     # Hybrid mode only
│       └── models/                          # Trained models (Phase 5)
│           └── xgboost.pkl
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

**Note**: All scripts automatically compute `project_root` as `Path(__file__).parent.parent`, pointing to the root directory. The default `projects_dir = "./projects"` correctly resolves to the projects folder.

## Installation

### Prerequisites

- Python 3.8 or higher
- Git (for repository cloning and commit analysis)
- C compiler (required for tree-sitter compilation)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/ECE4010.git
cd ECE4010

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt
```

### Optional: CodeBERT Support

For enhanced semantic features using CodeBERT embeddings:

```bash
pip install transformers torch
```

**Note**: GPU acceleration is recommended for CodeBERT inference.

## Quick Start

Here's a minimal example to get you started with a single project:

```bash
# Step 1: Static analysis on your project
python data/static_code_analysis.py \
    --source-dir projects/YourProject/source_code \
    --project-name YourProject \
    --projects-dir projects

# Step 2: Extract function features
python data/function_feature_extraction.py \
    --project YourProject \
    --projects-dir projects

# Step 3: Generate training labels (static mode, no git history needed)
python data/label_generation.py \
    --project YourProject \
    --projects-dir projects \
    --mode static \
    --negative-ratio 2.5

# Step 4: Construct pairwise features
python data/pairwise_features.py \
    --project YourProject \
    --projects-dir projects

# Step 5: Train the model
python train/train_xgboost.py \
    --project YourProject \
    --projects-dir projects

# Step 6: Predict co-change relationships
python verification/predict_cochange_graph.py \
    --project YourProject \
    --projects-dir projects \
    --model-path projects/YourProject/models/xgboost.pkl \
    --threshold 0.7 \
    --max-distance 2 \
    --output-format graphml \
    --output projects/YourProject/cochange_pred_graph
```

## Usage Guide

### Phase 0: Repository Setup (Optional)

If you want to use **hybrid mode** (leveraging commit history), first clone the target repository:

```bash
python data/data_cleaning.py \
    --repo-url https://github.com/username/repository.git \
    --repo-name YourProject \
    --max-commits 5000
```

**Output**:
- `projects/YourProject/source_code/` - Cloned source code
- `projects/YourProject/commits.json` - Commit history with modified functions

**Note**: Skip this phase if using **static mode** only.

### Phase 1: Static Code Analysis

Parse source code and build call graphs using Tree-sitter:

```bash
python data/static_code_analysis.py \
    --source-dir projects/YourProject/source_code \
    --project-name YourProject \
    --projects-dir projects
```

**Output**:
- `projects/YourProject/static_analysis/call_graph.pkl` - NetworkX directed call graph
- `projects/YourProject/static_analysis/function_metadata.json` - Function metadata archive

**Supported Languages**: Python, Java, JavaScript, TypeScript, C, C++, Go, Rust, and more (via Tree-sitter).

### Phase 2: Function Feature Extraction

Extract static features for each function:

```bash
python data/function_feature_extraction.py \
    --project YourProject \
    --projects-dir projects
```

**Output**:
- `projects/YourProject/features/function_features.npy` - Feature matrix (num_functions × num_features)
- `projects/YourProject/features/function_ids.json` - Function ID mapping

**Feature Categories**:
- Structural: Cyclomatic complexity, nesting depth, lines of code
- Syntactic: Parameter count, return type, function signature patterns
- Relational: Call graph centrality, fan-in/fan-out metrics

### Phase 3: Label Generation

Generate training labels indicating which function pairs co-occur in commits.

#### Option A: Hybrid Mode (Recommended)

Uses real commit history for accurate labels:

```bash
python data/label_generation.py \
    --project YourProject \
    --projects-dir projects \
    --mode hybrid \
    --min-cochange 2 \
    --negative-ratio 2.5 \
    --max-commits 2000
```

**Parameters**:
- `--min-cochange`: Minimum number of joint modifications to consider as positive sample (default: 2)
- `--negative-ratio`: Ratio of negative to positive samples (default: 2.5)
- `--max-commits`: Maximum commits to analyze (default: all)

#### Option B: Static Mode

Generates labels based on code structure alone (no git history):

```bash
python data/label_generation.py \
    --project YourProject \
    --projects-dir projects \
    --mode static \
    --negative-ratio 2.5
```

**Use Case**: Ideal for new projects without commit history or proprietary codebases.

**Output** (both modes):
- `projects/YourProject/labels/training_labels_v2.json` - Complete training labels
- `projects/YourProject/labels/positive_pairs.json` - Positive sample pairs
- `projects/YourProject/labels/negative_pairs.json` - Negative sample pairs
- `projects/YourProject/labels/commit_to_functions.json` - Commit-to-function mapping (hybrid only)

### Phase 4: Pairwise Feature Construction

Combine individual function features into pairwise features for classification:

```bash
python data/pairwise_features.py \
    --project YourProject \
    --projects-dir projects
```

**Output**:
- `projects/YourProject/features/X_train.npy`, `y_train.npy` - Training set
- `projects/YourProject/features/X_val.npy`, `y_val.npy` - Validation set
- `projects/YourProject/features/X_test.npy`, `y_test.npy` - Test set
- `projects/YourProject/features/scaler.pkl` - Feature standardizer (StandardScaler)

**Feature Combination Strategy**:
- Concatenation: `[feat_A, feat_B]`
- Absolute difference: `|feat_A - feat_B|`
- Element-wise product: `feat_A * feat_B`

### Phase 5: Model Training

Train an XGBoost classifier on the constructed features:

```bash
python train/train_xgboost.py \
    --project YourProject \
    --projects-dir projects
```

**Output**:
- `projects/YourProject/models/xgboost.pkl` - Trained XGBoost model

**Evaluation Metrics** (printed after training):
- Classification Report (Precision, Recall, F1-Score)
- AUC-ROC (Area Under ROC Curve)
- AUC-PR (Area Under Precision-Recall Curve)
- Confusion Matrix

**Hyperparameters** (configurable in script):
- `max_depth`: 6
- `learning_rate`: 0.1
- `n_estimators`: 200
- `scale_pos_weight`: Automatically calculated for class imbalance

### Phase 6: Inference & Prediction

Apply trained models to predict co-change relationships on new projects:

```bash
python verification/predict_cochange_graph.py \
    --project NewProject \
    --projects-dir projects \
    --model-path projects/TrainedProject/models/xgboost.pkl \
    --threshold 0.7 \
    --max-distance 2 \
    --output-format graphml \
    --output projects/NewProject/cochange_pred_graph
```

**Parameters**:
- `--model-path`: Path to trained XGBoost model (can be from different project)
- `--threshold`: Probability threshold for adding edges (default: 0.7)
- `--max-distance`: Maximum call graph distance for candidate pair generation (default: 2)
- `--output-format`: `graphml` (for Gephi) or `json` (for programmatic access)
- `--output`: Output file path (without extension)

**Output**:
- `projects/NewProject/cochange_pred_graph.graphml` - Prediction graph with weighted edges
- Edge weights represent predicted co-change probability

**Visualization**: Open the `.graphml` file in [Gephi](https://gephi.org/) or load with NetworkX for analysis.

## Multi-Project Joint Training

To improve cross-project generalization, we recommend **multi-project joint training**:

### Why Joint Training?

Single-project models often overfit to project-specific patterns. Joint training forces the model to learn **universal coupling patterns**, significantly improving transferability to unseen projects.

### Workflow

#### Step 1: Generate Labels for Multiple Projects

```bash
# For each project, generate labels
python data/label_generation.py --project ProjectA --projects-dir projects --mode static
python data/label_generation.py --project ProjectB --projects-dir projects --mode hybrid
python data/label_generation.py --project ProjectC --projects-dir projects --mode static
```

#### Step 2: Extract Function Features

```bash
python data/function_feature_extraction.py --project ProjectA --projects-dir projects
python data/function_feature_extraction.py --project ProjectB --projects-dir projects
python data/function_feature_extraction.py --project ProjectC --projects-dir projects
```

#### Step 3: Extract Pairwise Features

```bash
# Extract pairwise features for each project
python data/pairwise_features.py --project ProjectA --projects-dir projects
python data/pairwise_features.py --project ProjectB --projects-dir projects
python data/pairwise_features.py --project ProjectC --projects-dir projects
```

#### Step 4: Train Joint Model

Use the joint training script to train on multiple projects:

```bash
python train/train_joint.py \
    --projects ProjectA ProjectB ProjectC \
    --projects-dir projects \
    --output-dir model
```

**Output**:
- `model/xgboost_joint.pkl` - Joint-trained model
- `model/scaler_joint.pkl` - Global feature scaler
- `model/joint_training_metadata.json` - Training metadata

#### Step 5: Evaluate Cross-Project Generalization

```bash
python verification/evaluate_cross_project.py \
    --source "ProjectA+ProjectB+ProjectC" \
    --target NewProject \
    --projects-dir projects \
    --threshold 0.5 \
    --model-path model/xgboost_joint.pkl \
    --scaler-path model/scaler_joint.pkl
```
---

**Acknowledgments**: This project leverages several open-source libraries including [XGBoost](https://xgboost.ai/), [Tree-sitter](https://tree-sitter.github.io/), [NetworkX](https://networkx.org/), and [PyDriller](https://pydriller.readthedocs.io/).