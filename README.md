# Spam Detection: Naive Bayes vs Logistic Regression vs LinearSVC

A small machine learning project that classifies emails as spam or ham (not spam) using scikit-learn. I trained three models on the same data and compared them. The best one scores 99% on the dataset's test set, but flags 62% of my own real inbox as spam, so most of this project is about why that happens.

## Dataset

- Source: [Spam Email Dataset on Kaggle](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset)
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

## Testing on my own inbox

The test set above comes from the same dataset as the training data, so it can't show how the models behave on mail they have never seen anything like. I exported my own Gmail inbox (Google Takeout), converted it to CSV in the same `text` / `spam` format, and ran all three trained models on it with no retraining.

All 107 emails are real mail, so every email a model flags as spam is a false alarm.

| Model | Real emails flagged as spam | Rate |
|---|---|---|
| Naive Bayes | 1 of 107 | 0.9% |
| Logistic Regression | 41 of 107 | 38% |
| LinearSVC | 66 of 107 | 62% |

What this shows:

- **The best model on the benchmark is the worst on real mail.** LinearSVC made 2 false alarms in 872 real emails on the original test set, but it flags 62% of my inbox.
- **This is distribution shift.** The training emails look like older corporate mail, so the models appear to have learned "looks like that = ham, anything else = spam". Modern newsletters, receipts and other-language mail don't look like that.
- **Naive Bayes only looks good here because it is cautious.** It missed more than half the spam on the original test set, so a low false alarm rate says little about it.

Caveats: 107 emails is a small sample, so the exact percentages aren't reliable, but the gap between the benchmark and the real inbox is large. My inbox has no spam in it (I delete it), so this test only measures false alarms, not missed spam.

## Limitations

- **99% on the benchmark does not carry over to real mail.** See the inbox test above. The dataset's emails appear to be older corporate email (names, topics and dates inside some messages point to the early 2000s).
- **No modern spam to test or train on yet.** My inbox test covers false alarms only. I need recent spam from the same source as the ham to measure missed spam and to retrain fairly.
- **Single split.** Results come from one train/test split. Cross-validation would give a more reliable estimate.
- **Default TF-IDF.** No tuning of n-grams, stop words or other features.

## Next steps

- Collect modern spam (leave my Gmail spam folder alone for a few weeks, then export it), label it 1 and my inbox 0, and retrain on data from the same time and source. Then compare against the 62% false alarm rate above
- Check how many known words each flagged email has, to see whether the models are just guessing on text outside their vocabulary
- Add cross-validation and decision-threshold tuning
- Try character n-grams and extra features such as message length and number of links
- Build a small script that checks Gmail and only labels suspected spam, never deletes
