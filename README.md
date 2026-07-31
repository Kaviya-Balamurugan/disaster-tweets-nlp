# Disaster Tweets Classification — TF-IDF vs DistilBERT

Binary NLP classification of tweets as **real disaster** or **non-disaster** using the Kaggle *Natural Language Processing with Disaster Tweets* dataset.

This project compares a classical machine-learning NLP pipeline using **TF-IDF + Logistic Regression** with a pretrained Transformer model, **DistilBERT**. Additional experiments investigate keyword metadata and validation-based decision-threshold tuning.

---

## Problem Statement

Given a tweet, predict whether it describes a real disaster.

- `0` — Non-disaster
- `1` — Real disaster

The primary evaluation metric is **F1 score**, since both precision and recall are important for this binary classification task and F1 is also the Kaggle competition metric.

---

## Dataset

Dataset: Kaggle — *Natural Language Processing with Disaster Tweets*

The training dataset contains **7,613 labelled tweets**.

Available columns:

- `id` — Unique tweet identifier
- `keyword` — Disaster-related keyword, when available
- `location` — User-provided location, when available
- `text` — Tweet content
- `target` — Binary class label

### Target Distribution

| Class | Count | Percentage |
|---|---:|---:|
| Non-disaster | 4,342 | 57.03% |
| Disaster | 3,271 | 42.97% |

### Missing Values

- `text`: 0%
- `keyword`: 61 missing values (~0.8%)
- `location`: 2,533 missing values (~33.3%)

The dataset therefore has a moderate class imbalance, while location information is missing for approximately one-third of the training examples.

---

## Exploratory Data Analysis

EDA included:

- Target-class distribution
- Missing-value analysis
- Tweet character and word counts
- Duplicate tweet detection
- Conflicting-label analysis
- Examples from both classes

The average tweet lengths were:

- Non-disaster: **95.71 characters**
- Disaster: **108.11 characters**

The dataset also contains duplicate tweet texts. Some identical texts have conflicting target labels, indicating label ambiguity/noise in the dataset.

These observations were considered during validation and error analysis.

---

## Validation Strategy

A fixed **80/20 held-out validation split** was used for model development.

This produced:

- Training samples: **6,083**
- Validation samples: **1,530**

Identical tweet texts were grouped during splitting to reduce the risk of the same text appearing in both training and validation sets.

The same validation split was used when comparing the classical and Transformer approaches to make the results directly comparable.

---

## 1. Classical NLP Approach

The classical pipeline used:

**Tweet Text → TF-IDF → Logistic Regression**

Several configurations were evaluated:

- Unigram TF-IDF + Logistic Regression
- Unigram + Bigram TF-IDF + Logistic Regression
- Unigram TF-IDF + Balanced Logistic Regression
- Unigram + Bigram TF-IDF + Balanced Logistic Regression

Adding bigrams did not improve validation performance.

Using balanced class weights improved disaster recall and F1 compared with the initial Logistic Regression baseline.

### Selected Classical Model

**TF-IDF Unigram + Balanced Logistic Regression**

| Metric | Score |
|---|---:|
| Accuracy | 0.7987 |
| Precision | 0.7822 |
| Recall | 0.7544 |
| F1 Score | **0.7681** |

The classical model provides a fast, interpretable and computationally inexpensive baseline.

---

## 2. Transformer Approach — DistilBERT V1

The Transformer approach fine-tuned:

**`distilbert-base-uncased`**

V1 used only the raw tweet text as input.

### Training Configuration

- Model: DistilBERT
- Maximum sequence length: 128
- Learning rate: `2e-5`
- Training batch size: 16
- Evaluation batch size: 32
- Weight decay: 0.01
- Maximum epochs: 3
- Model selection metric: Validation F1

The best validation performance occurred at **epoch 2**.

### V1 Validation Results

| Metric | Score |
|---|---:|
| Accuracy | **0.8327** |
| Precision | **0.8420** |
| Recall | **0.7648** |
| F1 Score | **0.8016** |

### Confusion Matrix

| | Predicted Non-Disaster | Predicted Disaster |
|---|---:|---:|
| Actual Non-Disaster | 757 | 97 |
| Actual Disaster | 159 | 517 |

DistilBERT substantially outperformed the classical TF-IDF baseline, demonstrating the benefit of contextual language representations for disaster-tweet classification.

---

## 3. DistilBERT V2 — Keyword + Tweet Text

A second Transformer experiment investigated whether the provided `keyword` feature could improve classification.

Instead of using only:

```text
tweet text
```

V2 constructed the model input as:

```text
keyword [SEP] tweet text
```

For example:

```text
airplane accident [SEP] Experts in France begin examining...
```

Encoded spaces such as `%20` in keywords were converted to normal spaces.

When the keyword was missing, only the original tweet text was used.

### Sequence-Length Analysis

After adding keyword metadata:

- Mean sequence length: ~35.7 tokens
- Maximum sequence length: 84 tokens
- Sequences exceeding 128 tokens: 0

Therefore, the existing maximum sequence length of 128 remained sufficient.

### V2 Validation Results

| Metric | Score |
|---|---:|
| Accuracy | 0.8307 |
| Precision | 0.8283 |
| Recall | **0.7781** |
| F1 Score | **0.8024** |

Compared with V1, adding keyword metadata produced a small increase in validation F1 and recall, while precision and accuracy decreased slightly.

This indicates that the keyword feature provided some useful signal, but the improvement was marginal.

---

## 4. Decision-Threshold Tuning

The default binary decision threshold is approximately `0.50`.

To investigate whether the precision-recall trade-off could be improved, thresholds from **0.20 to 0.80** were evaluated on the held-out validation set.

The best threshold in the final reproducible V2 experiment was:

**0.54**

### Tuned V2 Results

| Metric | Score |
|---|---:|
| Accuracy | **0.8346** |
| Precision | **0.8395** |
| Recall | 0.7737 |
| F1 Score | **0.8052** |

### Tuned Confusion Matrix

| | Predicted Non-Disaster | Predicted Disaster |
|---|---:|---:|
| Actual Non-Disaster | 754 | 100 |
| Actual Disaster | 153 | 523 |

Threshold tuning improved validation F1 from **0.8024 to 0.8052**.

The threshold was selected using only the held-out validation data rather than the Kaggle leaderboard.

---

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| TF-IDF + Balanced Logistic Regression | 0.7987 | 0.7822 | 0.7544 | 0.7681 |
| DistilBERT V1 — Text Only | 0.8327 | 0.8420 | 0.7648 | 0.8016 |
| DistilBERT V2 — Keyword + Text | 0.8307 | 0.8283 | 0.7781 | 0.8024 |
| DistilBERT V2 + Threshold Tuning | **0.8346** | 0.8395 | **0.7737** | **0.8052** |

The Transformer models clearly outperform the classical TF-IDF baseline.

V2 and threshold tuning produced small improvements on the held-out validation set, although the gains were modest.

---

## Error Analysis

Error analysis was performed for both the classical and Transformer models.

The TF-IDF model often struggled with:

- Figurative uses of disaster-related words
- Tweets requiring contextual interpretation
- Ambiguous disaster terminology
- Informal social-media language

For example, words such as *ablaze*, *bomb*, *burning*, *attack*, and *annihilation* can describe either real events or figurative situations.

DistilBERT corrected many of these errors because it considers surrounding context rather than relying primarily on lexical features.

On the common validation split, DistilBERT corrected many predictions that the TF-IDF model classified incorrectly. However, the Transformer still struggled with ambiguous examples, noisy labels, and tweets whose meaning required additional external context.

---

## Kaggle Submission

After validation, the selected Transformer configuration was retrained on the complete **7,613-row training dataset** and used to predict the **3,263-row test dataset**.

### Public Leaderboard Results

The highest Kaggle public score was achieved by the **text-only DistilBERT V1 submission**:

**Kaggle Public F1 Score: 0.83757**

Later experiments using keyword metadata and threshold tuning improved held-out validation performance slightly but did not improve the public leaderboard score.

Therefore, the V1 text-only DistilBERT submission was retained as the **best Kaggle submission**.

This illustrates an important model-development consideration: small improvements on a single validation split do not always generalize to unseen data.

---

## Final Conclusion

The project demonstrates the progression from a classical NLP baseline to a contextual Transformer model.

The classical TF-IDF + Logistic Regression approach achieved an F1 score of **0.7681**, while text-only DistilBERT improved validation F1 to **0.8016**.

Adding keyword metadata increased validation F1 slightly to **0.8024**, and validation-based threshold tuning further increased it to **0.8052**.

However, these small validation improvements did not translate into better Kaggle public performance. The highest public leaderboard score remained **0.83757**, achieved by the text-only DistilBERT model.

The results demonstrate both the advantage of contextual Transformer representations over TF-IDF and the importance of separating validation-based model development from external leaderboard evaluation.

---

## Repository Structure

```text
Disaster-tweets-NLP/
│
├── notebooks/
│   ├── disaster_tweets_analysis.ipynb
│   └── distilbert_disaster_tweets.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

### Notebooks

**`disaster_tweets_analysis.ipynb`**

Contains:

- Data loading
- Exploratory data analysis
- Duplicate and conflicting-label analysis
- Validation strategy
- TF-IDF feature extraction
- Logistic Regression experiments
- Classical model evaluation
- Error analysis

**`distilbert_disaster_tweets.ipynb`**

Contains:

- DistilBERT tokenization
- Transformer fine-tuning
- Validation evaluation
- Confusion matrices
- Error analysis
- TF-IDF vs DistilBERT comparison
- Keyword metadata experiment
- Threshold tuning
- Final training
- Kaggle submission generation

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- Matplotlib
- Jupyter Notebook
- Kaggle GPU
- Git / GitHub

---

## Reproducibility

The classical experiments can be executed locally using the dependencies listed in `requirements.txt`.

Transformer fine-tuning was performed using a Kaggle GPU.

A fixed random seed and common validation split were used during model comparison to make the experiments as reproducible and directly comparable as possible.

---

## Key Results

**Best Classical Validation F1:** 0.7681  
**DistilBERT V1 Validation F1:** 0.8016  
**Best Validation F1:** 0.8052  
**Best Kaggle Public F1:** 0.83757

## AI Usage

AI tools (ChatGPT) were used as an assistant for code guidance, debugging, experiment planning, interpretation of results, and documentation.

All code was executed and verified by the candidate. Model training, validation, error analysis, experiment comparison, Kaggle submissions, and final model selection were performed and reviewed by the candidate.