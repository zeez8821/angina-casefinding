"""
Angina case-finding in untreated adults: research demonstration (Streamlit)

Runs the 9-predictor gradient boosting model developed and internally validated in UK Biobank
adults receiving neither cholesterol-lowering nor blood-pressure medication (JCDD submission,
analysis run full03). Output: flagged / not flagged at the pre-specified threshold (80%
sensitivity in out-of-fold training predictions). No individual risk percentage is shown,
because the model's predicted probabilities were poorly calibrated (calibration slope 0.56).

Research demonstration only. Not a medical device. Not validated for clinical use.
"""
import os

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Fixed settings from the analysis (do not edit)
# ---------------------------------------------------------------------------
MODEL_FILE = "final_model_untreated_selected"   # PyCaret pipeline + model (.pkl), run full03
THRESHOLD = 0.0018093121762992983                 # 80% sensitivity in training data
PREDICTORS = ["age", "ethnic", "chest_pain", "fam_heart_disease", "hba1c", "hdl",
              "urinary_sodium", "hematocrit", "red_blood_cell_count"]
# Allowed input ranges: 0.5th to 99.5th percentile of the untreated UK Biobank participants
# with complete data (default = median)
RANGES = {
    "age": (40, 69, 55),
    "hba1c": (24.4, 54.8, 34.5),
    "hdl": (0.77, 2.72, 1.44),
    "urinary_sodium": (12.5, 219.9, 66.8),
    "hematocrit": (31.6, 49.9, 41.0),
    "red_blood_cell_count": (3.52, 5.62, 4.49),
}
ETHNIC = ["White European", "South Asian", "Black Caribbean", "Black African", "Chinese",
          "Mixed", "Other ethnic group"]

# Test-set performance of this model at the threshold (untreated UK Biobank adults)
TEST_N, TEST_CASES = 63975, 214
TEST_SENSITIVITY = 0.804
TEST_SPECIFICITY = 0.690
TEST_FLAGGED = 0.312
TEST_FLAGGED_PER_CASE = 116
BASELINE_ONE_IN = 299

st.set_page_config(page_title="Angina case-finding: research demonstration", layout="centered")


@st.cache_resource
def load():
    from pycaret.classification import load_model
    here = os.path.dirname(os.path.abspath(__file__))
    return load_model(os.path.join(here, MODEL_FILE), verbose=False)


def flag(values: dict) -> bool:
    """True if the model flags the person at the pre-specified threshold."""
    row = pd.DataFrame([values])[PREDICTORS]
    return bool(load().predict_proba(row)[:, 1][0] >= THRESHOLD)


st.title("Angina case-finding")
st.caption("Research demonstration of a machine-learning model developed in UK Biobank")

st.warning(
    "**Research demonstration only.** This tool is not a medical device, has not been "
    "externally validated and must not be used for clinical decisions. It was developed and "
    "internally validated in UK Biobank participants, who are healthier than the general "
    "population.")

st.markdown(
    "**Who the model is for:** adults aged 40 to 69 years who are **not** taking "
    "cholesterol-lowering medication **and not** taking blood-pressure medication. "
    "The model indicates whether a person should be **flagged for clinical assessment** "
    "for possible angina. It does not diagnose angina.")

on_treatment = st.radio(
    "Is the person taking cholesterol-lowering or blood-pressure medication?",
    ["No", "Yes"], horizontal=True,
    help="Only these two types of medicine were used to define treatment in the study. People "
         "taking other heart medicines only (for example, aspirin or other antiplatelet drugs, "
         "or nitrates) were counted as untreated.")

if on_treatment == "Yes":
    st.info("The model was developed only in adults taking neither medication, so it "
            "cannot be applied to this person.")
    st.stop()

with st.form("inputs"):
    st.subheader("Questionnaire")
    c1, c2 = st.columns(2)
    age = c1.number_input("Age (years)", min_value=RANGES["age"][0],
                          max_value=RANGES["age"][1], value=RANGES["age"][2], step=1)
    ethnic = c2.selectbox("Ethnic group", ETHNIC, index=0,
                          help="Ethnic groups as recorded in UK Biobank.")
    chest = c1.radio("Do you ever have any pain or discomfort in your chest?", ["No", "Yes"],
                     horizontal=True, help="UK Biobank touchscreen question (field 2335).")
    fam = c2.radio("Family history of heart disease (parent or sibling)?", ["No", "Yes"],
                   horizontal=True, help="Heart disease reported for the father, mother or a "
                                         "brother or sister.")

    st.subheader("Blood and urine tests")
    c3, c4 = st.columns(2)
    hba1c = c3.number_input("HbA1c (mmol/mol)", min_value=RANGES["hba1c"][0],
                            max_value=RANGES["hba1c"][1], value=RANGES["hba1c"][2], step=0.1)
    hdl = c4.number_input("HDL cholesterol (mmol/L)", min_value=RANGES["hdl"][0],
                          max_value=RANGES["hdl"][1], value=RANGES["hdl"][2], step=0.01)
    hct = c3.number_input("Haematocrit (%)", min_value=RANGES["hematocrit"][0],
                          max_value=RANGES["hematocrit"][1], value=RANGES["hematocrit"][2],
                          step=0.1)
    rbc = c4.number_input("Red blood cell count (×10¹² cells/L)",
                          min_value=RANGES["red_blood_cell_count"][0],
                          max_value=RANGES["red_blood_cell_count"][1],
                          value=RANGES["red_blood_cell_count"][2], step=0.01)
    sodium = c3.number_input("Urinary sodium, spot urine (mmol/L)",
                             min_value=RANGES["urinary_sodium"][0],
                             max_value=RANGES["urinary_sodium"][1],
                             value=RANGES["urinary_sodium"][2], step=0.1)
    st.caption("Input ranges are limited to the range seen in the data used to develop the "
               "model (0.5th to 99.5th percentile).")
    submitted = st.form_submit_button("Check")

if submitted:
    flagged = flag({
        "age": float(age), "ethnic": ethnic,
        "chest_pain": 1.0 if chest == "Yes" else 0.0,
        "fam_heart_disease": 1.0 if fam == "Yes" else 0.0,
        "hba1c": float(hba1c), "hdl": float(hdl), "urinary_sodium": float(sodium),
        "hematocrit": float(hct), "red_blood_cell_count": float(rbc)})
    if flagged:
        st.error("### Flagged for clinical assessment")
    else:
        st.success("### Not flagged")
    st.markdown(
        f"**How to read this result.** In the UK Biobank test data (untreated adults), the "
        f"model flagged {TEST_FLAGGED:.0%} of people. About 1 in {TEST_FLAGGED_PER_CASE} "
        f"flagged people had diagnosed angina, compared with 1 in {BASELINE_ONE_IN} overall. "
        f"The model identified {TEST_SENSITIVITY:.0%} of people with diagnosed angina and "
        f"did not flag {TEST_SPECIFICITY:.0%} of people without it. A result of "
        f"\"not flagged\" does not rule out angina.")

with st.expander("About the model"):
    st.markdown(
        "- **Data:** UK Biobank baseline assessment (2006 to 2010); outcome: self-reported "
        "doctor-diagnosed angina.\n"
        "- **Population:** adults receiving neither cholesterol-lowering nor blood-pressure "
        f"medication (training 149,304; test {TEST_N:,}; {TEST_CASES} with diagnosed angina in "
        "the test set). People taking other heart medicines only (for example, antiplatelet "
        "drugs or nitrates) were counted as untreated.\n"
        "- **What the model was developed on:** undiagnosed angina cannot be observed in UK "
        "Biobank, so the model was developed with people who already had a doctor's diagnosis "
        "of angina but were not taking these medicines, as the closest available stand-in. "
        "How well it finds people with undiagnosed angina is unknown.\n"
        "- **Model:** gradient boosting; 9 predictors chosen from 37 by backward elimination "
        "with the one-standard-error rule, using training data only.\n"
        "- **Threshold:** chosen to give 80% sensitivity in the training data. Test-set AUC "
        "0.814 (95% CI 0.782 to 0.849).\n"
        "- **Chest pain:** chest pain was the most important predictor. People who know they "
        "have angina may report chest pain more often, which may make chest pain look more "
        "useful than it would be in people without a diagnosis.\n"
        "- **Differences between groups:** at the same threshold, the model flagged a larger "
        "share of men than women and of South Asian than White European participants. Few "
        "participants from other ethnic groups had diagnosed angina, so performance in these "
        "groups is uncertain.\n"
        "- **Limitations:** internal validation only; predicted probabilities are not shown "
        "because they were poorly calibrated and should not be read as individual risks.")
