# Angina case-finding: research demonstration (Streamlit)

Runs the 9-predictor gradient boosting model (age, ethnicity, chest pain, family history of heart
disease, HbA1c, HDL cholesterol, urinary sodium, haematocrit, red blood cell count) for adults receiving neither cholesterol-lowering nor
blood-pressure medication (UK Biobank; JCDD resubmission). Output: flagged / not flagged at the
pre-specified 80%-sensitivity threshold. No risk percentage is shown.

Files
- `app.py`: the app
- `final_model_untreated_selected.pkl`: fitted PyCaret pipeline + model from run full03 (no participant
  data), SHA-256 49bc1878...6bab1902. Threshold 0.0018093121762992983 (80% sensitivity in training).
  Checked: the app's flags equal the analysis pipeline's flags; test-set results in the paper:
  AUC 0.814, sensitivity 0.804, specificity 0.690, 31.2% flagged.
- `requirements.txt`: pycaret 3.3.2, streamlit 1.39.0
- `.streamlit/config.toml`: hides the developer toolbar, light theme

Deploy on Streamlit Community Cloud (new address)
1. Put these files (including the `.pkl` and the `.streamlit` folder) in a GitHub repository (a new one,
   or the one behind `cardiopredictsimple.streamlit.app`, replacing the old files). Main file: `app.py`.
   Choose a custom address when deploying, e.g. `angina-casefinding-ukb.streamlit.app`.
2. In the app's settings on share.streamlit.io, set **Python 3.10 or 3.11** (PyCaret 3.3.2 does not
   support newer versions). Changing the Python version requires deleting and redeploying the app
   with the same address.
3. Streamlit rebuilds the app from `requirements.txt` (the first build takes several minutes).

Run locally:  `pip install -r requirements.txt && streamlit run app.py`
