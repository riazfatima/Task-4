# Sentiment-Classification-Pipeline

## An End-to-End NLP Benchmarking Pipeline for Unstructured Text Polarity

**Project Status:** Mastery Milestone Completed

### The Core Problem: Taming the Invisible 80%

A fundamental challenge for modern enterprises is that 80% of data exists in unstructured, qualitative formats—such as product reviews, emails, and support tickets. This data is invisible to traditional analytics until it crosses the engineering threshold into quantitative spatial representations.

I built this project to solve that exact problem. My mandate was to engineer a strict, automated, multi-stage machine learning assembly line capable of ingesting raw, chaotic human language and generating high-accuracy probability distributions of emotional polarity.

### Engineering a Robust NLP Pipeline

This repository is built for production robustness, reproducibility, and clarity. It strictly follows a 5-step engineering lifecycle:

#### 1. Ingestion & Extraction

My pipeline dynamically manages local zip file extraction to structured data directories, handling potential character encoding issues automatically.

#### 2. Strict, POS-Guided Text Pre-Processing

Unprocessed text is full of syntactic noise, creates exponential redundancy, and leads to memory-based system crashes. To resolve this, I implemented three critical constraints:

**A. Character Normalization & Tokenization:** Standardization through regex-based URL removal, special character stripping, and strict lowercasing reduces the feature dimensions the model must learn.

**B. Resolving the Stop-Word Trap:** Default NLP dictionaries often contain negations. I implemented explicit, set-based union operations to ensure negations (like "not", "nor", "never") are preserved, preventing prediction failures on phrases like "not good".

**C. Accurate, Morphological Reduction (Lemmatization):** My pipeline uses NLTK's `WordNetLemmatizer`. To correct the pitfall where the lemmatizer defaults to noun reductions (e.g., failing to reduce "went" to "go"), I implemented POS tagging. My mandatory engineering rule is to map Penn Treebank tags to WordNet tags, ensuring accurate morphological reduction for verbs, adjectives, and nouns.

#### 3. Spatial Matrix Transformation (TF-IDF)

This step converts English text into mathematics.

- **Weight Calculation:** I implemented a combined TF (how often does the word appear in this document?) and IDF (how rare is the word across all documents?) logic to assign weights, ensuring non-sentiment fluff words receive low weights while rare, high-sentiment words (e.g., 'terrible') receive high weights.
- **N-Gram Capturing:** I included Bigrams (2-word relationships) to explicitly capture negated sentiment (e.g., 'not good'), while setting boundaries on `max_features` to manage vocabulary explosion and dimensionality issues.

#### 4. CSR Sparse Compression

Vectorizing 10,000 reviews against a vocabulary of 20,000 terms results in a matrix that is 99% empty zeros. Attempting to store this as a dense array exhausts RAM, causes fatalities on system crashes, and operates at an abysmal O(N³). Following best practices for training linear classifiers, my implementation compresses this void using SciPy's Compressed Sparse Row (CSR) format, ensuring industry-standard efficiency and memory usage.

#### 5. Comparative Modeling

With my text mathematically vectorized, I benchmarked two robust models:

- **Multinomial Naive Bayes:** A model that is best suited for perfectly balanced datasets and applies Laplace Smoothing to handle zero-frequency problems in testing.
- **Linear Support Vector Machine (SVC):** This classifier also proved effective in identifying polarity boundaries.

### Visualizing Mathematical Certainty

The pipeline dynamically generates and saves comparative visualizations, providing immediate clarity on results.

#### Sentiment Class Distribution

My script confirms a highly balanced distribution of sentiment in the training data, with slightly over 8,000 instances of each class (Positive and Negative). This balance prevents extreme bias during training.

| Model Performance Comparison Matrix (Confusion Matrices) | ROC Curve Comparison (Benchmarking Metrics) |
| --- | --- |
| ![Confusion Matrices](https://github.com/riazfatima/Task-4/blob/main/Model%20Performance%20Comparison%20Matrix.png?raw=true) | ![ROC Curves](https://github.com/riazfatima/Task-4/blob/main/ROC%20curve%20comparison.png) |

*Side-by-side matrices demonstrate exceptional accuracy for both models.* SVM correctly predicted **1,869** examples. NB slightly behind, with **1,862** examples.

*ROC benchmarking shows both models are exceptionally strong, achieving a matching AUC (Area Under Curve) of 0.96.* NB and SVM are virtually overlayed, demonstrating excellent separability.

### Final Metric Report

```text
============================================================
              NAIVE BAYES METRIC REPORT (Accuracy: 0.88)
============================================================
              precision    recall  f1-score   support

    Negative       0.89      0.87      0.88      1001
    Positive       0.88      0.90      0.89      1103

weighted avg       0.88      0.88      0.88      2104

============================================================
                SVM CLASS METRIC REPORT (Accuracy: 0.89)
============================================================
              precision    recall  f1-score   support

    Negative       0.88      0.89      0.88      1001
    Positive       0.90      0.89      0.89      1103

weighted avg       0.89      0.89      0.89      2104


