# Spam Detection Using Machine Learning

## Overview
This project builds a machine learning model to classify SMS messages as **spam** or **ham (not spam)** using natural language processing techniques. It demonstrates a complete pipeline from data preprocessing to model training and evaluation.

---

## Dataset
The dataset used is the **SMS Spam Collection** dataset, containing 5,572 messages labeled as spam or ham. It was sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).

---

## Features & Approach
- **Text Cleaning:** Converted text to lowercase, removed punctuation.
- **Vectorization:** Used TF-IDF with unigrams and bigrams to convert text into numeric features.
- **Modeling:** Trained a Logistic Regression classifier with balanced class weights to handle class imbalance.
- **Evaluation:** Achieved ~98% accuracy with strong precision and recall on spam detection.

---

## Results

| Metric    | Spam Class | Ham Class | Overall Accuracy |
|-----------|------------|-----------|------------------|
| Precision | 0.93       | 0.98      |                  |
| Recall    | 0.89       | 0.99      | 0.98             |
| F1-Score  | 0.91       | 0.99      |                  |

*Confusion matrix and classification report are available in the notebook.*

---

## How to Run
1. Clone this repository.
2. Upload the `spam.csv` dataset into the working directory.
3. Open the notebook `Spam_Detection.ipynb` in [Google Colab](https://colab.research.google.com) or locally with Jupyter Notebook.
4. Run each cell sequentially to reproduce the results.

---

## Tools & Libraries
- Python 3.x
- pandas
- scikit-learn
- matplotlib & seaborn
- Google Colab (for cloud execution)

---

## Future Improvements
- Experiment with other models like Random Forests or Gradient Boosting.
- Implement stemming or lemmatization to enhance text preprocessing.
- Build a simple web app interface to demo the spam detector.

---

## Contact
Feel free to reach out if you have questions or want to collaborate!

---

*This project is a great example of applying machine learning to solve real-world problems using industry-standard tools and best practices.*

