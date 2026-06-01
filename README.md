# NeoStats Credit Risk Platform

An end-to-end AI platform for predicting loan defaults, explaining decisions, and exploring banking data using natural language.

## Setup Instructions

### 1. Local Environment Setup
1. Create virtual environment: `python -m venv venv`
2. Activate it: `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Add your Gemini/OpenAI API key to `.env`

### 2. Prepare Data & Model
1. Place `application_train.csv` inside `/data`
2. Initialize DB and Train Model:
   ```bash
   python -m src.ml.train