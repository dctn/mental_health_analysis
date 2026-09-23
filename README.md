# Student Mental Health Analysis

Exploratory analysis of student depression survey data, plus a text classifier that labels a written statement with the mental-health state it expresses.

This is the code behind **"AI-Driven Early Detection of Student Mental Health Risks"**, awarded *Best Paper* at **AI-SDSC'25**, the National Conference on AI for Sustainable Development and Smart Cities (December 2025).

> ⚠️ Research and educational project. It is not a diagnostic tool and must not be used to assess, screen or advise a real person.

---

## What's in here

| Path | What it is |
|---|---|
| `student_mental_health.ipynb` | Exploratory data analysis of the student depression survey |
| `model training.ipynb` | Text preprocessing, vectorisation, model training and inference |
| `dataset/Student Depression Dataset.csv` | Survey data — 27,901 students, 18 columns |
| `dataset/student_mental_health_nlp/dataset.csv` | 53,043 written statements labelled with a mental-health status |
| `models/tf_idf.pkl` | Fitted TF-IDF vectoriser (2,000 features, 1–2 grams) |
| `models/logistic_regression_model.pkl` | Trained logistic-regression classifier |
| `models/label_encoder_for_target.pkl` | Label encoder for the seven classes |

---

## Part 1 — Exploratory analysis

27,901 student records across 18 columns: age, city, academic pressure, CGPA, study satisfaction, sleep duration, dietary habits, financial stress, family history of mental illness, and self-reported suicidal thoughts.

**What the data shows**

- **58.5%** of students in the dataset are labelled as depressed.
- **63.3%** report having had suicidal thoughts.
- **50.0%** report both — but **8.5%** are depressed *without* reporting suicidal thoughts, and a substantial group reports suicidal thoughts without a depression label. Self-report alone does not identify who is at risk.
- **Academic pressure** and **financial stress** show the strongest positive correlation with depression among the numeric columns.
- Students sleeping **under 5 hours** are disproportionately represented in the depressed group.
- Depressed students skew towards **lower CGPA** — the direction of cause is not established, and the notebook says so.

Cleaning was minimal: the dataset has no nulls, and the two yes/no columns (family history, suicidal thoughts) were mapped to 0/1. Plots use seaborn — count plots by gender, financial stress, academic pressure and sleep duration; KDE plots for age, CGPA and academic pressure; a correlation heatmap; and the ten cities with the most depressed students.

---

## Part 2 — Text classifier

Classifies a free-text statement into one of **seven classes**: `Anxiety`, `Bipolar`, `Depression`, `Normal`, `Personality disorder`, `Stress`, `Suicidal`.

**Pipeline**

1. Drop rows with a missing statement — 53,043 → **52,681** usable statements.
2. Clean each statement: strip non-alphabetic characters, lowercase, split, remove English stopwords, lemmatise (WordNet).
3. Vectorise. Bag-of-words (150 features) was tried first; **TF-IDF with 2,000 features and 1–2 grams** was kept → matrix of `(52681, 2000)`.
4. Encode the seven labels, then balance the classes with `RandomOverSampler`.
5. Train an 80/20 split on logistic regression and random forest.
6. Persist the vectoriser, model and label encoder with `joblib` for reuse.

**Scores**

| Model | Score |
|---|---|
| Random forest | 0.982 |
| Logistic regression | 0.827 |

Both numbers come from `.score()` on the **full resampled dataset**, which includes the rows the models trained on — so they measure fit, not generalisation. Treat them as such; see *Limitations* below.

**Inference**

The last section of the notebook takes a typed sentence, runs it through the same cleaning steps, loads the saved TF-IDF vectoriser and classifier, and prints the predicted class — a paragraph describing low mood and withdrawal comes back as `Depression`.

---

## Running it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn nltk joblib
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
jupyter notebook
```

Run `student_mental_health.ipynb` for the analysis, `model training.ipynb` to reproduce the classifier. The training notebook reads the statements dataset from a relative path — point that cell at `dataset/student_mental_health_nlp/dataset.csv` if you run it from the repository root.

---

## Limitations and next steps

Written down rather than left for a reader to find:

- **Oversampling happens before the train/test split**, so duplicated minority rows can appear on both sides. The reported scores are inflated by this. The fix is to split first, then resample the training half only.
- **No held-out evaluation yet** — no classification report, confusion matrix or per-class F1. With seven imbalanced classes, per-class recall matters far more than overall accuracy, especially for `Suicidal`.
- **Labels are self-reported categories**, not clinical diagnoses, and the two datasets come from different populations. Nothing here transfers to a clinical setting.
- **`Depression` and `Suicidal` overlap heavily in language**, which is exactly where a classifier's errors carry the most weight.
- Next steps: proper split-then-resample evaluation, per-class metrics, a calibrated probability output instead of a hard label, and a comparison against a transformer baseline.

---

## Datasets

- *Student Depression Dataset* — survey responses of Indian students.
- *Sentiment Analysis for Mental Health* — statements labelled by mental-health status.

Both are public datasets used here for research and coursework.
