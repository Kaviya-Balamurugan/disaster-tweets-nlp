# Disaster Tweets Classification — TF-IDF vs DistilBERT

Binary NLP classification of tweets as real disaster or non-disaster using the Kaggle Natural Language Processing with Disaster Tweets dataset.

The project compares a classical TF-IDF based approach with a pretrained Transformer model.

## Problem

Given the text of a tweet, predict:

- `0` — Non-disaster
- `1` — Real disaster

## Dataset

Kaggle: Natural Language Processing with Disaster Tweets

Training data contains 7,613 labelled tweets.

Features provided include:

- `text`
- `keyword`
- `location`
- `id`

The target column is `target`.

## Exploratory Analysis

The training set contains:

- 4,342 non-disaster tweets (57.03%)
- 3,271 disaster tweets (42.97%)

Missing values:

- `keyword`: 0.80%
- `location`: 33.27%
- `text`: 0%

Duplicate analysis found 69 unique duplicated tweet texts, including 18 texts associated with conflicting labels.

Disaster tweets were slightly longer on average, although the character-count distributions showed substantial overlap.

## Validation Strategy

A fixed 80/20 held-out validation set was used.

Identical tweet texts were grouped during splitting so the same text could not appear in both training and validation sets.

This reduced the risk of duplicate-text leakage.

The same validation strategy was used when comparing the classical and Transformer approaches.

## Classical Approach

TF-IDF features were evaluated with Logistic Regression.

Experiments included:

- Unigram TF-IDF + Logistic Regression
- Unigram + bigram TF-IDF
- Class-balanced Logistic Regression

Class weighting improved disaster recall and F1, while adding bigrams did not improve validation F1.

The selected classical model was:

**Unigram TF-IDF + Balanced Logistic Regression**

## Transformer Approach

`distilbert-base-uncased` was fine-tuned for binary sequence classification.

Configuration:

- Maximum sequence length: 128
- Learning rate: 2e-5
- Batch size: 16
- Weight decay: 0.01
- Maximum epochs: 3
- Best checkpoint selected using validation F1

The highest validation F1 occurred at epoch 2.

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| TF-IDF + Balanced Logistic Regression | 0.7987 | 0.7822 | 0.7544 | 0.7681 |
| DistilBERT | 0.8327 | 0.8420 | 0.7648 | 0.8016 |

DistilBERT achieved the strongest held-out validation performance.

Its confusion matrix contained:

- True negatives: 757
- False positives: 97
- False negatives: 159
- True positives: 517

Compared with the classical model, DistilBERT substantially reduced false positives while also slightly reducing false negatives.

Error analysis showed that DistilBERT corrected many TF-IDF errors involving contextual or figurative language. However, both approaches still struggled with ambiguous examples and noisy labels.

## Final Training and Kaggle Submission

Based on the validation experiment, two epochs were selected for final DistilBERT training.

The final model was retrained on all 7,613 labelled training examples and used to predict the 3,263 Kaggle test tweets.

**Kaggle Public Score: 0.83757**

## Repository Structure

```text
.
├── notebooks/
│   ├── disaster_tweets_analysis.ipynb
│   └── distilbert_disaster_tweets.ipynb
├── README.md
├── requirements.txt
└── .gitignore