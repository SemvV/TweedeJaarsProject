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

## Requirements
### Required python packages
- **General packages**  
  - Pandas
  - numpy
  - matplotlib
  - nltk
  - sklearn/ scikit-learn

- **Supervised labeling models**  
  - torch
  - transformers
  - tqdm
  - reportlab
  - requests

- **Non-supervised labeling models**  
  - sentence_transformers
  - bertopic

- **Violence prediction models**  
  - xgboost

### Runtime requirements
Below are the hardware/runtime requirements for training specific models. Anything besides the below mentioned topics, should be a able to run on a low-end cpu or gpu.
- **Supervised note labeling**  
  For the supervised note labeling models, we recommend using a gpu. Running on a CUDA this takes around 1.5h and with an Apple M4 it takes around 2.5h.

- **Semi-supervised note labeling**  
  A GPU is recommended. ESS takes up to 30 minutes running on a geforce rtx 3070, while ZSC takes around two hours to classify 77,496 notes.

- **Unsupervised note labeling**
Neither is recommended.

- **Binary violence prediction**  
  We recommend the binary violence prediction models to be run on google colab or a low-end gpu/ cpu. On google colab, running the four different binary prediction models will take around 10 to 15 minutes.

- **Probability violence prediction**  
  XXX

## File Overview

### Data Preparation
- **DataCleaning.py**  
  Preprocessing script for cleaning and filtering raw ACLED events. Generates `filtered_events_country_code.csv` in the `data/` folder.

### Embedding-Based Labeling
- **EmbeddingSimilaritySearch.ipynb**  
  Applies Embedding Similarity Search (ESS) to label event notes by comparing against predefined topic descriptions.

### Violence Prediction Models
- **violence_model_binary.ipynb**  
  Implements and evaluates scikit-learn models for binary violence prediction (violent vs. non-violent protests). Also includes the generation of the data needed and the
  factors most influencing violence according to the logistic regression model.
- **violence_orientation_binary.ipynb**  
  Explores feature engineering and orientation analysis for the binary violence prediction model (and violence prediction in general)
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
- **transformers_model_finetuner.py**  
  Script to train and evaluate a Hugging Face Transformer for text classification, with automated report generation (PDF) and Ollama-based model review.
- **labeling_by_transformer_model.ipynb**  
  An script to label the dataset with an pretrained transformer model

---