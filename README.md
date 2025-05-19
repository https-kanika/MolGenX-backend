# MolGenX Backend

Backend service for the MolGenX drug discovery platform that handles molecule generation, optimization, and protein-ligand interaction analysis.

## Prerequisites

- Python 3.11.9
- Virtual Environment

## Setup Instructions

### 1. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Activate virtual environment (Linux/MacOS)
source .venv/bin/activate

#Install Dependencies 
pip install -r requirements.txt

```

### Local Development

```bash

 python main.py

 # This will start server at http://localhost:8080

```

### API Endpoints

- POST: /api/optimize
    
    Request Body: 
    ```json
    {
        "protein": "SEQUENCE" or "pdb_id": "ID",
        "weights": {
            "druglikeness": 1.0,
            "synthetic_accessibility": 0.8,
            "lipinski_violations": 0.7,
            "toxicity": 1.2,
            "binding_affinity": 1.5,
            "solubility": 0.6
        }
    }
    ```
    