# Student guide: demo, requirement checklist and viva preparation

## A. Demonstration sequence (about 5 minutes)
1. `python train_model.py` - show the six-model table and "Best model by F1-score: Gradient Boosting".
2. `python test_pipeline.py` - show all checks passing.
3. `streamlit run app.py` - open the browser page.
4. Predict with the default values (no failure expected), then Type L / 1250 RPM / 70 Nm / 240 min (failure expected).
5. Enter a negative torque to show validation. Open the "Model information" tab to show metrics and limitations.
6. Take screenshots into `screenshots/` and insert one into Figure 10 of the report.

## B. Requirement checklist (honest status)
| Requirement | Implemented | In report | Tested | Demonstrated |
|---|---|---|---|---|
| Deploy best model | Yes (app.py + saved pipeline) | Sec. 6.5, 7.4 | Logic tested; real Streamlit UI NOT run yet | Pending - you |
| Demonstrate deployed model | Instructions in README / section A | Fig. 10 placeholder | No | Pending - you |
| Complete Python code | Yes | - | Training and 9 checks run | - |
| 1 Problem statement & objective | - | Yes | - | - |
| 2 Metadata | - | Yes (verified vs CSV) | Yes | - |
| 3 EDA & preprocessing | Yes | Yes | Yes | - |
| 4 Metrics table | Yes | Yes | Reproduced exactly | - |
| 5 Plots | Yes (9 genuine plots) | Yes; Figure 10 screenshot missing | - | Pending - you |
| 6 Model comparison | Yes | Yes | - | - |
| 7.1 AI declaration with prompts | - | Phase 3 filled; Phase 1/2 and full prompt need YOU | - | - |
| 7.2 Critical evaluation | - | Yes | - | - |
| 7.3 Reflection | - | Draft only; your answers needed | - | - |
| 7.4 Design ownership | - | Yes; confirm column needed | - | - |

## C. Things only you can finish
- Run the app on your laptop and add the screenshot; fill the submission date.
- Section 7.1: tools/prompts for Phases 1-2, and paste the full Phase 3 prompt as an appendix.
- Section 7.3: answer the four highlighted questions in your own words. Section 7.4: tick the confirm column.
- Re-run `notebooks/phase2.ipynb` and compare Decision Tree / Random Forest with the report (see report 7.2).
- Check the references (they were written from memory, not looked up).

## D. Viva preparation
**Why Gradient Boosting?** Highest F1 (0.7563) and accuracy (0.9855); precision 0.8824, recall 0.6618. Logistic Regression has higher recall (0.8235) but precision 0.1418 (339 false alarms). Random Forest has higher precision but recall 0.4265.
**What does the pipeline do?** ColumnTransformer: numbers -> median impute -> StandardScaler; Type -> most-frequent impute -> OneHotEncoder. Then the classifier. All in one Pipeline, fitted on training data only.
**How does the app load the model?** `joblib.load("models/best_model_pipeline.pkl")` once (cached). No training in app.py.
**How do inputs become a prediction?** Validate -> one-row DataFrame with the six training columns in order -> `pipeline.predict` / `predict_proba` -> show class and probability (failure if probability >= 0.5).
**Metrics:** accuracy = correct/all; precision = real failures among predicted failures; recall = detected among real failures; specificity = correct among normals; F1 = harmonic mean of precision and recall.
**Training vs deployment:** training fits and saves (once, offline); deployment only loads and predicts new inputs.
**Limitations:** synthetic data; misses 23 of 68 test failures; small number of failures; model chosen on the test set (optimistic); no tuning/CV; prototype only.
**Likely questions:** Why exclude TWF/HDF/PWF/OSF/RNF? (leakage). Why stratify? (keep 3.4% failures in both sets). Why not accuracy? (96.6% baseline). Why is the file a Pipeline, not just a model? (so scaling/encoding match). What if Type is unseen? (handle_unknown=ignore, but app only allows L/M/H). How could recall improve? (class weights, threshold tuning, CV - not done here). Why does the Phase 2 notebook output differ for DT/RF? (not determined - re-run it).
**Responsible AI:** Phase 3 code/report were AI-assisted and declared; results were reproduced and checked; unfinished parts are marked.
