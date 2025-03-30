# Spam-sms-detection-using-ML
Develop and SMS classification model that identifies the spam messages

# 📌 Spam SMS Detection using Machine Learning

## 📜 Project Overview
This project aims to develop an SMS classification model that identifies spam messages. The dataset consists of labeled messages categorized as spam or non-spam. The goal is to build a robust model that accurately distinguishes between spam and legitimate messages using machine learning techniques.

## 📂 Dataset
- **Dataset Name:** SMS Spam Collection
- **Source:** [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data)
- **Description:** A dataset containing 5,574 SMS messages labeled as either spam or ham (legitimate).
- **Format:** The dataset consists of two columns:
  - `v1`: Label ("ham" for legitimate messages, "spam" for spam messages)
  - `v2`: The raw SMS text

## 🔍 Tasks Performed
1. **Data Loading & Exploration**
   - Read and analyze the dataset
   - Display class distribution (spam vs. ham)
   - Identify missing values and handle inconsistencies
   
2. **Data Preprocessing**
   - Convert labels into numerical values (`ham = 0`, `spam = 1`)
   - Remove special characters, punctuation, and extra spaces
   - Perform tokenization, stopword removal, and stemming
   
3. **Exploratory Data Analysis (EDA)**
   - Visualize class distribution using bar charts
   - Analyze message length and word count differences between spam and ham messages
   - Generate word clouds to display frequently used words
   
4. **Feature Engineering**
   - Extract text features using **TF-IDF vectorization**
   - Compute correlation between features

5. **Model Training & Evaluation**
   - Train multiple classification models:
     - **Naïve Bayes**
     - **Logistic Regression**
     - **Support Vector Machine (SVM)**
     - **K-Nearest Neighbors (KNN)**
     - **Decision Tree**
     - **Random Forest**
     - **XGBoost**
   - Evaluate models using:
     - **Accuracy, Precision, Recall, and F1-score**
     - **Confusion matrix**
     - **Cross-validation for model stability**
   - **Best Model:** SVM performed the best in terms of accuracy and overall performance



## 🛠 Installation & Usage
### 🔹 Install Dependencies
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud xgboost joblib etc 
```

### 🔹 Run the Project
```bash
Spam sms detection.ipynb
just open the file and see the code 
```


## 📊 Results & Observations
- **Spam messages are typically longer and contain more numbers.**
- **Word clouds show distinct vocabulary differences between spam and ham messages.**
- **SVM performed best among all tested models.**

## 🚀 Future Improvements
- Apply **deep learning models** (LSTMs, Transformers) for better accuracy.
- **Hyperparameter tuning** to further optimize model performance.
- **Deploy as an interactive web application.**


