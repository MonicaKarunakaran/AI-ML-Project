# RAGAS / Edge-Case Analysis Report

## 1. Purpose

This document records the limitations, edge cases, and potential failure scenarios
identified for the W4D5 Sentiment Classifier.

The main classifier uses TF-IDF features with Logistic Regression and Random Forest.

---

## 2. Positive and Negative Sentiment

The model performs binary sentiment classification:

- 0 → Negative
- 1 → Positive

The dataset should contain reasonably balanced examples from both classes.

---

## 3. Potential Edge Cases

### Very Short Text

Examples:

- "Good"
- "Bad"
- "Okay"

Very short inputs contain limited information and may be harder to classify.

### Negation

Examples:

- "I did not like this."
- "This is not good."

A simple bag-of-words representation may not fully understand the relationship
between negation and sentiment.

### Sarcasm

Examples:

- "Great, another broken update."
- "Amazing service, waited for two hours."

Sarcasm is difficult for traditional TF-IDF based models because the literal
words may have a different meaning from the intended sentiment.

### Mixed Sentiment

Example:

"I liked the design, but the product was extremely slow."

The text contains both positive and negative opinions.

### Out-of-Domain Text

The model may perform poorly on text that is significantly different from
the training data.

---

## 4. Bias Considerations

The model can inherit biases from the training dataset.

Potential sources include:

- Uneven class distribution
- Repeated examples
- Domain-specific vocabulary
- Dataset-specific writing styles
- Limited representation of different types of users

---

## 5. Model Limitations

TF-IDF does not understand language in the same way as modern transformer models.

It mainly represents text using word importance and frequency.

Therefore, the model may have difficulty with:

- Sarcasm
- Context
- Negation
- Long-range relationships
- Unseen vocabulary

---

## 6. Future Improvements

Possible improvements include:

1. Use a larger and more diverse dataset.
2. Tune Logistic Regression and Random Forest hyperparameters.
3. Add cross-validation.
4. Experiment with word and character n-grams.
5. Compare against a transformer-based sentiment model.
6. Monitor model performance after deployment.
7. Add human feedback for incorrect predictions.

---

## 7. RAGAS Note

RAGAS is primarily designed for evaluating RAG systems involving retrieved
contexts and generated answers.

This project is a traditional supervised classification pipeline, so RAGAS is
not the primary evaluation framework for the classifier itself.

The project instead documents model quality, edge cases, and limitations.