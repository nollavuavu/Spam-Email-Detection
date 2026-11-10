I have decided my first AI/ML project to be spam detection.

# Spam Detection: Naive Bayes vs Logistic Regression vs LinearSVC

A small machine learning project that classifies emails as spam or ham (not spam) using scikit-learn. I trained three models on the same data and compared them.

## Dataset

- 5,728 labeled emails, two columns: the email text and a label (1 = spam, 0 = ham)
- About 24% spam and 76% ham, so the classes are imbalanced
- No missing values and no duplicate rows
- Because of the imbalance, a model that always says "ham" would already score 76% accuracy. Accuracy alone is misleading here, so I focus on spam precision and spam recall.

## Method

1. Stratified train/test split: 80% train, 20% test (1,146 test emails), `random_state=42`
2. `TfidfVectorizer` with default settings, fitted on the training set only and then applied to the test set
3. Trained three classifiers on the same TF-IDF features
4. Evaluated on the held-out test set with `classification_report` and `confusion_matrix`

## Results

Spam class (label 1):

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Multinomial Naive Bayes | 1.00 | 0.44 | 0.61 |
| Logistic Regression (`class_weight="balanced"`) | 1.00 | 0.91 | 0.95 |
| LinearSVC (`class_weight="balanced"`) | 0.99 | 0.99 | 0.99 |

Overall accuracy: 0.87 (NB), 0.98 (LR), 0.99 (LinearSVC).

Confusion matrices, as `[[ham correct, ham flagged as spam], [spam missed, spam caught]]`:

| Model | Real ham flagged as spam | Spam missed | Spam caught |
|---|---|---|---|
| Naive Bayes | 0 | 154 | 120 |
| Logistic Regression | 0 | 26 | 248 |
| LinearSVC | 2 | 4 | 270 |

### What the numbers say

- **Naive Bayes** never flagged a real email, but it missed 154 of 274 spam emails. It leans toward the majority class (ham) when it is unsure.
- **Logistic Regression** with balanced class weights fixed most of that: recall went from 0.44 to 0.91 with no false alarms.
- **LinearSVC** was the best overall. It made only 6 mistakes in 1,146 test emails (2 real emails flagged, 4 spam missed). It traded 2 false alarms for 22 fewer missed spam compared to Logistic Regression.

## Error analysis

I read all 6 emails LinearSVC got wrong.

- **Real emails flagged as spam (2):** one about price-matching clauses and one about a billing/credit card problem. Both use vocabulary that spam also uses (price, offer, credit card).
- **Spam missed (4):** one was random word salad, one looked like an ordinary out-of-office autoreply (possibly a labeling error), and two were mass-mailing style messages written in a calm, business-like tone.

All of these are borderline cases, and some may be label noise. The model is close to the ceiling for this dataset.

## Limitations

- **99% is probably too optimistic.** The emails look like older corporate email (names, topics and writing style suggest a 2000s-era corpus), so the model may partly be learning "sounds like old corporate email" versus everything else.
- **Not tested on modern mail.** Newsletters, receipts, promotions, other languages and current phishing look different. Expect lower performance on a real inbox (distribution shift).
- **Single split.** Results come from one train/test split. Cross-validation would give a more reliable estimate.
- **Default TF-IDF.** No tuning of n-grams, stop words or other features.

## Next steps

- Test the saved model on my own inbox (all real mail, so every flag is a false positive) to measure the distribution shift
- Add cross-validation and decision-threshold tuning
- Try character n-grams and extra features such as message length and number of links
- Build a small script that checks Gmail and only labels suspected spam, never deletes