
# 📧 UniMail – Intelligent Email Classification System

An intelligent email classification system built using Machine Learning, Natural Language Processing (NLP), and the Gmail API. UniMail automatically categorizes university-related emails into meaningful categories, helping students organize important academic and campus communications.

## 🚀 Project Overview

University students receive a large volume of emails related to examinations, assignments, placements, scholarships, hackathons, seminars, and other campus activities.

Manually managing these emails can be time-consuming and may cause students to miss important announcements.

**UniMail** addresses this problem through an automated email classification pipeline that analyzes email subjects and message bodies, predicts their categories using trained machine learning models, and presents the results through an interactive Streamlit application.

## ✨ Key Features

- **Multi-Class Email Classification:** Automatically categorizes university emails into 10 predefined categories.
- **Machine Learning Model Comparison:** Evaluates multiple classification algorithms to identify the best-performing model.
- **NLP-Based Feature Extraction:** Uses TF-IDF vectorization with unigram and bigram features.
- **Gmail Integration:** Fetches emails through the Gmail API using OAuth 2.0 authentication.
- **Interactive Dashboard:** Displays categorized emails through a Streamlit interface.
- **Category-Wise Email Browsing:** Filter and explore emails based on their predicted categories.
- **Email Analytics:** Displays category distribution and email statistics.
- **CSV Export:** Download classified email results for further analysis.
- **Verified Label Training:** Uses manually verified email labels to improve classification quality.

## 📂 Email Categories

UniMail currently supports the following 10 categories:

| Category | Description |
|---|---|
| Hackathons | Coding competitions and hackathon announcements |
| Exams | Examination schedules and academic notices |
| Hostel Notices | Hostel-related announcements |
| Seminars | Technical talks, workshops and seminars |
| Sports | Sports events and competitions |
| Scholarships | Scholarship opportunities and announcements |
| Assignments | Assignments, submissions and academic tasks |
| Events | University events and activities |
| General Announcements | General university communications |
| Placements | Placement opportunities and recruitment updates |

## 🧠 Machine Learning Methodology

The classification pipeline uses the following steps:

1. **Data Preparation:** Collect and organize university-related email subjects and bodies.
2. **Label Verification:** Review email categories and use verified labels for model training.
3. **Text Processing:** Combine email subjects and message bodies into a single text input.
4. **Feature Extraction:** Apply TF-IDF vectorization using unigram and bigram features.
5. **Model Training:** Train and compare multiple machine learning classifiers.
6. **Model Evaluation:** Evaluate models using stratified cross-validation.
7. **Model Selection:** Select the best-performing model based on macro F1-score.
8. **Model Deployment:** Save the selected pipeline using Joblib for application-level predictions.

### Models Implemented

| Algorithm | Purpose |
|---|---|
| Multinomial Naive Bayes | Probabilistic text classification |
| Logistic Regression | Multi-class classification |
| Linear Support Vector Machine | Linear classification with margin optimization |

### Evaluation Metrics

The models are evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Classification Report
- Confusion Matrix

Evaluation results are generated dynamically during model training and saved in the `outputs/` directory.

## 🖥️ Application

The Streamlit application provides an interactive interface for connecting Gmail and exploring classified emails.

### Gmail Integration

- Google OAuth 2.0 authentication.
- Read-only Gmail API permissions.
- Fetch recent emails from an authorized Gmail account.
- Extract email sender, date, subject and body.
- Automatically predict email categories.
- Browse classified messages.
- Export classification results as CSV.

**Privacy:** The application requests read-only Gmail access. It does not modify, delete, send or create labels for Gmail messages.

## 🛠️ Technology Stack

| Technology | Usage |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| Scikit-learn | Machine learning algorithms |
| TF-IDF | Text feature extraction |
| Joblib | Model serialization |
| Streamlit | Interactive web application |
| Gmail API | Email retrieval |
| Google OAuth 2.0 | Authentication |
| BeautifulSoup | HTML email content extraction |
| Matplotlib / Seaborn | Data visualization, where used |

## 📁 Project Structure

```text
Email_Classification/
│
├── data/
│
├── models/
│   ├── best_email_classifier.joblib
│   └── email_categories.txt
│
├── outputs/
│   ├── ml_labeling_queue.xlsx
│   ├── model_comparison.csv
│   ├── best_model_classification_report.csv
│   ├── best_model_confusion_matrix.csv
│   └── best_model_cv_predictions.csv
│
├── src/
│   ├── auto_categorize.py
│   ├── check_body_integrity.py
│   ├── create_labeling_queue.py
│   ├── create_validation_sample.py
│   ├── inspect_data.py
│   ├── prepare_data.py
│   └── sample_emails.py
│
├── app.py
├── demo.py
├── check_accuracy.py
├── replace_labeling_queue_with_real_emails.py
├── train_campusmail_models.py
├── unimail_streamlit_app.py
│
├── test_gmail_connection.py
├── test_gmail_classification.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Singh1veer/Email_Classification.git

cd Email_Classification
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
source .venv/Scripts/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📩 Gmail API Configuration

To connect Gmail:

1. Open the Google Cloud Console.
2. Create or select a Google Cloud project.
3. Enable the Gmail API.
4. Configure the OAuth consent screen.
5. Create OAuth 2.0 credentials for a Desktop application.
6. Download the OAuth credentials JSON file.
7. Rename it to `credentials.json`.
8. Place it in the project root directory.
9. Add your Google account as a test user if the OAuth application is in testing mode.

The application uses the following read-only permission:

```python
https://www.googleapis.com/auth/gmail.readonly
```

The first authentication generates `token.json`, which stores the authorization token locally.

**Important:** Never upload `credentials.json` or `token.json` to GitHub.

## 🏋️ Model Training

Ensure the verified email-labeling workbook is available at:

```text
outputs/ml_labeling_queue.xlsx
```

Run the training pipeline:

```bash
python train_campusmail_models.py
```

The training script:

- Loads manually verified email labels.
- Validates the ten supported categories.
- Extracts TF-IDF features.
- Trains three machine learning classifiers.
- Performs stratified cross-validation.
- Compares model performance.
- Selects the best-performing classifier.
- Saves the final trained pipeline.

The selected model is saved as:

```text
models/best_email_classifier.joblib
```

## ▶️ Run the Application

Launch the Streamlit application:

```bash
streamlit run unimail_streamlit_app.py
```

Open the local URL displayed in your terminal to access UniMail.

## 📊 Model Evaluation Outputs

The training pipeline generates:

| File | Purpose |
|---|---|
| model_comparison.csv | Comparison of all trained models |
| best_model_classification_report.csv | Precision, recall and F1-score |
| best_model_confusion_matrix.csv | Actual versus predicted categories |
| best_model_cv_predictions.csv | Cross-validation predictions |
| best_email_classifier.joblib | Selected trained classification pipeline |

## 🔐 Security and Privacy

- Gmail access is restricted to read-only OAuth permissions.
- OAuth credentials and access tokens should remain private.
- Sensitive email data should not be committed to public repositories.
- Model predictions should be manually verified for important communications.

## 🔮 Future Enhancements

- Fraudulent placement and internship email detection.
- Scholarship announcement fraud detection.
- Phishing and suspicious email identification.
- Improved classification through larger verified datasets.
- Enhanced email analytics and filtering.
- Further experimentation with advanced NLP and machine learning models.

## 👨‍💻 Author

**Harshveer Singh**

GitHub: [Singh1veer](https://github.com/Singh1veer)

---

*UniMail – Making university email management smarter through Machine Learning and NLP.*
