# Language Identification System

A small language identification project that classifies short text samples as **English**, **Swahili**, **Sheng**, or **Luo**.

## What It Does

- Loads text samples from the four language files
- Cleans and preprocesses the text
- Trains and compares multiple models
- Saves the best model and evaluation artifacts
- Provides a Streamlit app for live predictions

## Project Files

- [main.py](main.py) - training, evaluation, and model saving
- [app.py](app.py) - Streamlit user interface
- [requirements.txt](requirements.txt) - Python dependencies
- [english.txt](english.txt), [swahili.txt](swahili.txt), [sheng.txt](sheng.txt), [luo.txt](luo.txt) - source text samples

## Requirements

- Python 3.12+
- The packages in [requirements.txt](requirements.txt)

## Setup

Install dependencies inside the project virtual environment:

```bash
./venv/bin/pip install -r requirements.txt
```

If you are not using the bundled virtual environment, install the packages in your own environment instead.

## Train the Model

Run the training script to build the dataset, compare models, and save the trained bundle:

```bash
./venv/bin/python main.py
```

This creates:

- [language_dataset.csv](language_dataset.csv)
- [artifacts/model_comparison.csv](artifacts/model_comparison.csv)
- [artifacts/confusion_matrix.png](artifacts/confusion_matrix.png)
- [artifacts/language_identifier.joblib](artifacts/language_identifier.joblib)

## Launch the Streamlit App

After training, start the interface with:

```bash
./venv/bin/streamlit run app.py
```

If Streamlit is available on your PATH, this also works:

```bash
streamlit run app.py
```

## When You Add More Data

If you add more samples to the same four language files, retraining is usually all you need.

Recommended workflow:

1. Update [english.txt](english.txt), [swahili.txt](swahili.txt), [sheng.txt](sheng.txt), or [luo.txt](luo.txt).
2. Retrain with [main.py](main.py):

```bash
./venv/bin/python main.py
```

3. Restart Streamlit if it is running, so [app.py](app.py) loads the updated model.

When retraining alone is not enough:

1. If you add a new language class, also update the language mapping in [main.py](main.py).
2. If your data format changes (for example, no longer one sample per line), update the loading logic in [main.py](main.py).


## Expected Usage

1. Paste a short text sample into the input box.
2. Click **Predict language**.
3. Read the predicted language from the result panel.

## Notes

- The dataset is balanced across all four languages before training.
- Sheng text is lightly normalized to handle common slang.
- Character n-grams work best for this task, so the final model uses a Linear SVM pipeline.
