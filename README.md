
# ⚽ Football Expected Goals (xG) Analytics

An interactive football analytics dashboard that estimates the probability of a shot resulting in a goal using machine learning. Built with Python, XGBoost, Scikit-learn, Streamlit, and StatsBomb open event data.

## 🚀 Features

- **Expected Goals Prediction:** Estimates goal probability for individual shots.
- **XGBoost Model:** Trains a gradient-boosted tree classifier.
- **Shot Context Analysis:** Uses shot location, distance, angle, goalkeeper distance, and defensive pressure features.
- **Interactive Dashboard:** Explores shot distributions, shot maps, and team and player statistics where available.
- **Custom Shot Predictor:** Adjust shot and defensive-context inputs to explore predicted probabilities.
- **Model Evaluation:** Compares predictive performance using ROC-AUC, log loss, and Brier score.

## 📊 Model Performance

Evaluated on a held-out set of 76 matches, with matches separated between training and testing.

| Metric | XGBoost |
|---|---:|
| ROC-AUC | 0.7903 |
| Log Loss | 0.2746 |
| Brier Score | 0.0777 |

A higher ROC-AUC is better. Lower log loss and Brier score are better. These results represent one held-out split and do not guarantee performance on other competitions or seasons.

## 🛠️ Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Plotly
- Joblib

## 📁 Project Structure

```text
Football-xG/
├── app.py
├── train_model.py
├── validate_model.py
├── compare_models.py
├── evaluate_context_model.py
├── train_xgboost.py
├── predict_xg.py
├── extract_context_features.py
├── inspect_freeze_frames.py
├── data/
│   ├── shots.csv
│   ├── shots_with_context.csv
│   └── events/
├── xg_model.joblib
├── xgboost_model.joblib
├── calibration.png
├── .gitignore
└── README.md
```

The exact files in your local folder may differ. Upload only files that exist and are needed.

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Football-xG
```

### 2. Create and activate a virtual environment

Windows:

```bash
py -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install pandas numpy scikit-learn xgboost streamlit plotly joblib
```

### 4. Start the dashboard

```bash
streamlit run app.py
```

Ensure the dataset and saved XGBoost model are present in their expected locations.

## 📚 Dataset and Attribution

This project uses StatsBomb open event data. Data availability and reuse are subject to the [StatsBomb Open Data repository](https://github.com/statsbomb/open-data) and its attribution requirements. Consult the repository's current terms before redistributing the raw dataset.

## ⚠️ Limitations

- Evaluation uses a single held-out match split.
- Results may vary across competitions, seasons, and playing styles.
- Freeze-frame context features depend on the available event data.
- The custom predictor uses user-supplied defensive-context estimates, which may not represent a real match situation.
- Predicted xG is a statistical estimate, not a guarantee of scoring.

## 🔮 Future Improvements

- Evaluate with multiple match-based splits or cross-validation.
- Improve probability calibration and test on additional competitions.
- Add richer player and team comparisons.
- Deploy the dashboard online.
- Explore additional spatial and contextual features.

## 👤 Author

**Mohammed Usman**  
GitHub: [@Usmantech678](https://github.com/Usmantech678)
