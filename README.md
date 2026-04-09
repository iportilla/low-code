# Sample Prompts for Low-Code Classification Lab

This document contains sample prompts that can be used with an LLM (like GPT-4o or Claude) to generate the scripts for the Lost & Found classification project.

---

## 1. Traditional ML Classifier (`classifier.py`)

**Prompt:**
> Write a Python script named `classifier.py` that builds a traditional machine learning pipeline for text classification.
> 
> **Requirements:**
> - **Dataset:** Load `lost-50.csv`. The input text is in the `Ticket` column and the target label is in the `Category` column.
> - **Preprocessing:** Use a Scikit-Learn `Pipeline` with `TfidfVectorizer` (use unigrams and bigrams, and log-scaled frequency).
> - **Model:** Use `LogisticRegression` with `class_weight='balanced'` to handle the small and imbalanced dataset.
> - **Evaluation:** Perform 5-fold cross-validation and print the mean accuracy.
> - **Saving:** Save the entire trained pipeline to `models/tfidf_logreg.joblib` using the `joblib` library.
> - **CLI:** When run directly, training should execute and display a final test split accuracy (using an 80/20 split).

---

## 2. LLM-Based Classifier (`classifier_llm.py`)

**Prompt:**
> Write a Python script named `classifier_llm.py` that implements zero-shot classification using the OpenAI API.
> 
> **Requirements:**
> - **API:** Use the `openai` library and `gpt-4o-mini` model. Load the API key from a `.env` file using `python-dotenv`.
> - **Logic:**
>     - Extract the list of unique categories from `lost-50.csv`.
>     - Construct a system prompt that provides these categories to the LLM and instructs it to only return the exact category name.
>     - Implement a function `classify_item(description)` that calls the API.
> - **Saving Config:** Save the list of categories and the system prompt used to `models/llm_classifier_config.json`.
> - **Evaluation Mode:** Add a CLI flag `--evaluate`. If present, the script should run classification on the first 20 items of `lost-50.csv` and report the overall accuracy.
> - **Interactive Mode:** If run without flags, prompt the user for an item description and print the predicted category.

---

## 3. Streamlit Application (`app.py`)

**Prompt:**
> Write a Streamlit application named `app.py` to provide a user interface for the Lost & Found classifiers.
> 
> **Requirements:**
> - **Sidebar:** Create a model selector to choose between "TF-IDF + Logistic Regression" and "GPT-4o-mini".
> - **Input:** Provide a text area for the user to enter a lost item description (e.g., "black leather wallet").
> - **Logic:**
>     - If the ML model is selected, load `models/tfidf_logreg.joblib` and display the predicted category, a confidence score, and a bar chart of the top 5 class probabilities.
>     - If the LLM model is selected, use the logic from `classifier_llm.py` to predict the category. Display the result along with the API latency.
> - **Extra Feature:** For both models, if a prediction is made, search the `lost-50.csv` file for existing items in that same category and display them in a table to show "similar found items".
> - **Styling:** Use a clean layout with appropriate headers and metric cards for performance stats.

---

## 4. Dependencies (`requirements.txt`)

**Prompt:**
> Create a `requirements.txt` file for this project. It should include:
> - `pandas` (for data handling)
> - `scikit-learn` and `joblib` (for the ML classifier)
> - `openai` and `python-dotenv` (for the LLM classifier)
> - `streamlit` (for the web app)

---

## 5. Environment Setup (.env.example)

**Prompt:**
> Create a `env.example` file that contains a placeholder for the `OPENAI_API_KEY`.

---

## 6. Installation & Execution

**Prompt:**
> Provide the terminal commands to set up the environment and run the project.
> 
> **Steps:**
> - Creation of a virtual environment named `.venv`.
> - Activation of the virtual environment.
> - Installation of dependencies using `pip install -r requirements.txt`.
> - Commands to run `classifier.py`, `classifier_llm.py`, and `streamlit run app.py`.
