# AI-Based Predictive Maintenance System (Phase 3)

MCA ML Lab mini project - AI4I 2020 Predictive Maintenance Dataset.
A Streamlit app that loads the best model from Phase 2 (Gradient Boosting) and predicts
machine failure from product type and operating conditions. Academic prototype, not production software.

## Folder structure
```
app.py                  Streamlit application (loads the saved model, never retrains)
train_model.py          Trains the six Phase 2 models, saves the best pipeline, metrics and plots
pipeline_utils.py       Shared features, preprocessing, models, metrics, input validation
test_pipeline.py        Automated checks for the workflow
requirements.txt        Dependencies
data/ai4i2020.csv       Dataset (already included)
models/                 best_model_pipeline.pkl + model_info.json (created by train_model.py)
notebooks/              phase1.ipynb, phase2.ipynb (original notebooks; only the CSV path changed to ../data/)
outputs/                model_comparison.csv and plots/ (created by train_model.py)
report/                 Phase 3 report (DOCX and PDF)
screenshots/            Put your app screenshots here
```

## Setup
Python 3.10 or newer (the code was tested with Python 3.12 and scikit-learn 1.8).
```
python -m venv venv
venv\Scripts\activate          (Windows)   |   source venv/bin/activate   (macOS/Linux)
pip install -r requirements.txt
```

## 1. Train and save the model
The dataset must be at `data/ai4i2020.csv` (it is included; if you replace it, keep this name and location).
```
python train_model.py
```
This prints the six-model comparison table, saves `models/best_model_pipeline.pkl` (preprocessing + model in one
pipeline), `models/model_info.json` (metrics shown in the app), `outputs/model_comparison.csv` and the plots.
Re-run it on your own laptop so the saved model matches your scikit-learn version.

## 2. Run the checks (optional but recommended)
```
python test_pipeline.py
```

## 3. Start the application
```
streamlit run app.py
```
Open the address shown in the terminal (normally http://localhost:8501).

## 4. Test a prediction
- Normal conditions: Type M, air 300.0 K, process 310.0 K, 1500 RPM, 40 Nm, 100 min -> no failure expected.
- High-stress conditions: Type L, 1250 RPM, 70 Nm, 240 min tool wear -> failure expected.
- Try a negative torque to see the validation message. Rename the .pkl file to see the missing-model message.

## Notes
- Results use the same split (80:20, random_state=42, stratified) and hyperparameters as Phase 2.
- The model file is trained on the 80% training split only; reported metrics come from the 20% test split.
- If loading fails with a version warning, run `python train_model.py` again.
