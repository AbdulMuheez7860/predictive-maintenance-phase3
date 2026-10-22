# AI-Based Predictive Maintenance System (Phase 3)

MCA ML Lab (MCA 5141) mini project - AI4I 2020 Predictive Maintenance Dataset.
A Streamlit app that loads the best model from Phase 2 (Gradient Boosting) and predicts machine failure
from product type and operating conditions. Academic prototype, not production software.

- **Live demo:** https://predictive-maintenance-phase3.streamlit.app/
- **Code:** https://github.com/AbdulMuheez7860/predictive-maintenance-phase3

## Results (test set, 2,000 records)
| Metric | Gradient Boosting |
|---|---|
| Accuracy | 0.9855 |
| Precision | 0.8824 |
| Recall | 0.6618 |
| Specificity | 0.9969 |
| F1-score | 0.7563 |

Gradient Boosting had the highest F1-score of the six models compared (Logistic Regression, Decision Tree, KNN,
Gaussian Naive Bayes, Random Forest, Gradient Boosting). It misses 23 of 68 failures in the test set.
The full comparison is written to `outputs/model_comparison.csv` when you run `train_model.py`.

## Folder structure
```
app.py                  Streamlit application (loads the saved model, never retrains)
train_model.py          Trains the six Phase 2 models, saves the best pipeline, metrics and plots
pipeline_utils.py       Shared features, preprocessing, models, metrics, input validation
test_pipeline.py        Automated checks for the workflow
requirements.txt        Dependencies (pinned versions)
data/ai4i2020.csv       Dataset (already included)
models/                 best_model_pipeline.pkl + model_info.json (created by train_model.py)
notebooks/              phase1.ipynb, phase2.ipynb (original notebooks; only the CSV path changed to ../data/)
outputs/                model_comparison.csv and plots/ (created by train_model.py)
report/                 Phase 3 report (DOCX and PDF) - local copy only, not uploaded to GitHub
screenshots/            App screenshots used in the report - local copy only
STUDENT_GUIDE.md        Demo steps, requirement checklist and viva preparation
```

## Setup
Python 3.10 or newer (tested with Python 3.12 / scikit-learn 1.8 and on a student laptop with scikit-learn 1.9.1).
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
Re-run it on your own machine so the saved model matches your scikit-learn version.

## 2. Run the checks (optional but recommended)
```
python test_pipeline.py
```
Expected: `9/9 checks passed`.

## 3. Start the application locally
```
streamlit run app.py
```
Open the address shown in the terminal (normally http://localhost:8501).

## 4. Test a prediction
- Normal conditions: Type L, air 299.9 K, process 309.8 K, 1500 RPM, 40.0 Nm, 100 min -> NO FAILURE, about 0.2% probability.
- High-stress conditions: Type L, 1250 RPM, 70 Nm, 240 min tool wear -> failure predicted.
- Try a negative torque to see the validation message. Rename the .pkl file to see the missing-model message.

## 5. Deploy online (Streamlit Community Cloud)
1. Push this project to a public GitHub repository (`venv/`, `report/` and `screenshots/` are listed in `.gitignore`).
2. Make sure `requirements.txt` uses exact versions (`==`) that match the version that trained the model.
3. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**, choose the repository,
   branch `main` and main file `app.py`, then **Deploy**.
4. If the app fails to load the model, check the logs (**Manage app**), match the scikit-learn version
   and Python version, and reboot the app.

## Notes
- Results use the same split (80:20, random_state=42, stratified) and hyperparameters as Phase 2.
- The model file is trained on the 80% training split only; reported metrics come from the 20% test split.
- The best model was chosen using the same test set that reports its metrics, so the scores may be slightly optimistic.
- The dataset is synthetic; predictions are estimates and must not be used for real maintenance decisions.
- If loading fails with a version warning, run `python train_model.py` again.
