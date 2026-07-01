import os
import re
import zipfile
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Graphic Packages
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure required NLTK resources are downloaded
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# ==========================================
# PHASE 1: INGESTION & EXTRACTION
# ==========================================

zip_path = r'c:\Users\NEW LAPTOP CITY\.vscode\Internship Projects\Project 4\archive (1).zip'
extract_dir = 'extracted_sentiment_data'

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extracted archive to: {extract_dir}")
else:
    extract_dir = '.' 

train_path = os.path.join(extract_dir, 'train.csv')
test_path = os.path.join(extract_dir, 'test.csv')

if not os.path.exists(train_path) and os.path.exists(os.path.join(extract_dir, 'training.1600000.processed.noemoticon.csv')):
    train_path = os.path.join(extract_dir, 'training.1600000.processed.noemoticon.csv')
    test_path = os.path.join(extract_dir, 'testdata.manual.2009.06.14.csv')

print(f"Loading Training Data from: {train_path}")
print(f"Loading Testing Data from: {test_path}")

try:
    df_train = pd.read_csv(train_path, encoding='ISO-8859-1')
    df_test = pd.read_csv(test_path, encoding='ISO-8859-1')
except Exception:
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

def prepare_columns(df):
    if df.shape[1] == 6:
        df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    elif 'text' in df.columns and 'sentiment' in df.columns:
        pass 
    else:
        df = df.rename(columns={df.columns[0]: 'sentiment', df.columns[-1]: 'text'})
    
    df = df[['text', 'sentiment']].dropna().reset_index(drop=True)
    df['sentiment'] = df['sentiment'].replace({4: 1, 'positive': 1, 'positive ': 1, 0: 0, 'negative': 0, 'negative ': 0})
    df = df[df['sentiment'].isin([0, 1])].reset_index(drop=True)
    return df

df_train = prepare_columns(df_train)
df_test = prepare_columns(df_test)

print(f"Final Train Shape: {df_train.shape} | Final Test Shape: {df_test.shape}")

# --- NEW VISUALIZATION: Class Distribution Plot ---
plt.figure(figsize=(6, 4))
sns.countplot(x='sentiment', data=df_train, palette='viridis')
plt.title('Training Data: Sentiment Class Distribution')
plt.xticks([0, 1], ['Negative (0)', 'Positive (1)'])
plt.xlabel('Sentiment Class')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# ==========================================
# PHASE 2: STRICT TEXT PRE-PROCESSING
# ==========================================

base_stopwords = set(stopwords.words('english'))
negation_words = {'no', 'not', 'nor', 'neither', 'never', 'none', "isn't", "aren't", "wasn't", "weren't", 
                  "haven't", "hasn't", "hadn't", "won't", "wouldn't", "don't", "doesn't", "didn't", "can't", "cannot", "couldn't"}
custom_stopwords = base_stopwords - negation_words

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text_pipeline(text):
    text = re.sub(r'<[^>]+>', '', text)  
    text = re.sub(r'http\s+|www\S+|https\S+', '', text)  
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  
    text = re.sub(r'[^A-Za-z\s]', '', text)  
    text = text.lower()
    
    raw_tokens = text.split()
    
    # FIXED LOGIC: Run POS Tagging on the FULL structural sentence BEFORE dropping stop words
    pos_tags = nltk.pos_tag(raw_tokens)
    
    lemmatized_tokens = []
    for word, tag in pos_tags:
        if word in negation_words or word not in custom_stopwords:
            wordnet_tag = get_wordnet_pos(tag)
            root_word = lemmatizer.lemmatize(word, wordnet_tag)
            lemmatized_tokens.append(root_word)
            
    return " ".join(lemmatized_tokens)

print("Executing cleaning and pre-processing pipeline on datasets...")
df_train['cleaned_text'] = df_train['text'].apply(clean_text_pipeline)
df_test['cleaned_text'] = df_test['text'].apply(clean_text_pipeline)


# ==========================================
# PHASE 3: VECTORIZATION & TRANSFORMATION
# ==========================================

print("Transforming qualitative text into numeric sparse matrices via TF-IDF...")
vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1, 2))

X_train_sparse = vectorizer.fit_transform(df_train['cleaned_text'])
y_train = df_train['sentiment'].astype(int)

X_test_sparse = vectorizer.transform(df_test['cleaned_text'])
y_test = df_test['sentiment'].astype(int)


# ==========================================
# PHASE 4: CLASSIFICATION MODELING & EVALUATION
# ==========================================

# Model 1: Naive Bayes
print("\nTraining Model 1: Multinomial Naive Bayes...")
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_sparse, y_train)
y_pred_nb = nb_classifier.predict(X_test_sparse)
y_score_nb = nb_classifier.predict_proba(X_test_sparse)[:, 1]

# Model 2: Support Vector Machine (SVM)
print("Training Model 2: Support Vector Machine (Linear SVC)...")
svm_classifier = LinearSVC(random_state=42, dual='auto')
svm_classifier.fit(X_train_sparse, y_train)
y_pred_svm = svm_classifier.predict(X_test_sparse)
y_score_svm = svm_classifier.decision_function(X_test_sparse)

# Text Reports Output
print("\n" + "="*60)
print("             NAIVE BAYES METRIC REPORT")
print("="*60)
print(classification_report(y_test, y_pred_nb, target_names=['Negative', 'Positive']))

print("\n" + "="*60)
print("               SVM CLASS METRIC REPORT")
print("="*60)
print(classification_report(y_test, y_pred_svm, target_names=['Negative', 'Positive']))


# ==========================================
# PHASE 5: LIVE GRAPHICAL VISUALIZATION
# ==========================================
print("\nGenerating live side-by-side model comparison plot...")

cm_nb = confusion_matrix(y_test, y_pred_nb)
cm_svm = confusion_matrix(y_test, y_pred_svm)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Performance Comparison Matrix', fontsize=16, fontweight='bold')

# Plot Naive Bayes Matrix
sns.heatmap(cm_nb, annot=True, fmt='d', ax=axes[0], cmap='Blues',
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
axes[0].set_title('Model 1: Multinomial Naive Bayes', fontsize=12)
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

# Plot SVM Matrix
sns.heatmap(cm_svm, annot=True, fmt='d', ax=axes[1], cmap='Greens',
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
axes[1].set_title('Model 2: Support Vector Machine (SVM)', fontsize=12)
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

plt.tight_layout()
plt.show()

# --- NEW VISUALIZATION: ROC Curve Comparison ---
print("\nGenerating model ROC curve comparison plot...")
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_score_nb)
roc_auc_nb = auc(fpr_nb, tpr_nb)

fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

plt.figure(figsize=(8, 6))
plt.plot(fpr_nb, tpr_nb, color='blue', lw=2, label=f'Naive Bayes (AUC = {roc_auc_nb:.2f})')
plt.plot(fpr_svm, tpr_svm, color='green', lw=2, label=f'Linear SVM (AUC = {roc_auc_svm:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve Comparison')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()