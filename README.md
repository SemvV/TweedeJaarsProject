# Gaining New Insights into Protests Using AI

This project, conducted in collaboration with the Clingendael Institute, leverages artificial intelligence to gain deeper insights into the causes and dynamics of protests across Europe. Using a filtered version of the ACLED dataset, we developed models to categorize protest topics and predict the likelihood of violent escalation.

---

## Directory Structure
```
TweedeJaarsProject/                 # Root of the GitHub repository
├── data/                           # Folder for datasets or raw data files
└── TweedeJaarsProject/             # Main project folder
    └── supervised_transformer_labeling/  # Supervised transformer-based labeling models
``` 

Note: The `data/` folder is excluded from GitHub due to large file sizes.

---

## File Overview

### Data Preparation
- **DataCleaning.py**  
  Preprocessing script for cleaning and filtering raw ACLED events. Generates `filtered_events_country_code.csv` in the `data/` folder.

### Embedding-Based Labeling
- **EmbeddingSimilaritySearch.ipynb**  
  Applies Embedding Similarity Search (ESS) to label event notes by comparing against predefined topic descriptions.

### Violence Prediction Models
- **violence_model_binary.ipynb**  
  Implements and evaluates scikit-learn models for binary violence prediction (violent vs. non-violent protests).
- **violence_orientation_binary.ipynb**  
  Explores feature engineering, orientation analysis, and data preparation for the binary violence model.
- **violence_probability.ipynb**  
  Trains a logistic regression model to predict violence probability by actor and country. Usage:
  ```python
  print(predict_violence('country_code', 'actor_name'))
  ```

### Trigger-Word Labeling
- **labeling_triggerwords.ipynb**  
  Labels protests using a dictionary of trigger words. Customize classes and keywords in `Classes_dic` as needed.

### Unsupervised Topic Modeling
- **Unsupervised_LDA.ipynb**  
  Applies Latent Dirichlet Allocation (LDA) for unsupervised topic labeling.
- **Unsupervised_BERTopic.ipynb**  
  Uses BERTopic for unsupervised topic modeling of protest notes.

### Zero-Shot Classification
- **Zero-ShotClassification.ipynb**  
  Semi-supervised labeling using Zero-Shot classification with DeBERTa v3.
- **Zero-ShotClassificationTop3.ipynb**  
  Retrieves the top three most likely topics via Zero-Shot DeBERTa v3.

---

## Map: `supervised_transformer_labeling`

- **dataset_balancing.ipynb**  
  Addresses class imbalance by grouping smaller topic classes into broader categories.
- **supervised_labeling_bert.ipynb**  
  Introduction and tutorial for fine-tuning BERT on protest topic classification.
- **transformers_model_finetuner.py**  
  Script to train and evaluate a Hugging Face Transformer for text classification, with automated report generation (PDF) and Ollama-based model review.

---