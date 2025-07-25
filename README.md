# Fraudshield-online-detector
An AI-powered web application to detect online payment fraud using a machine learning model. Built with Flask, HTML/CSS, and scikit-learn.


Dataset Information
This project uses the PaySim simulated financial transaction dataset hosted on Kaggle:
Source: Kaggle – PaySim Dataset

Filename: PS_20174392719_1491204439457_log.csv
Size: ~470 MB
Records: 6.3 million
Target Variable: isFraud (1 = fraudulent, 0 = legitimate)
Imbalance: Only ~0.129% are fraudulent transactions

⚠️ Due to GitHub's 100MB file upload limit, this file is not included in the repository.
🔽 How to Download the Dataset
To run the project locally:

Visit the dataset page:
👉 https://www.kaggle.com/datasets/ealaxi/paysim1

Log in with your Kaggle account and download the ZIP.

Extract the contents and move the CSV file here:

Online-Fraud-Payment-Detector/

├── PS_20174392719_1491204439457_log.csv  ✅ <-- Required

├── app.py

├── train_model.py

├── fraud_model.pkl

├── scaler.pkl

├── templates/

└── index.html

├── static/

🧠 Features Used
Feature Name     	Description
step	            Hour of transaction from start of simulation
type	            Transaction type (CASH_OUT, TRANSFER, etc.)
amount	          Transaction amount
oldbalanceOrg     Sender balance before transaction
newbalanceOrig	  Sender balance after transaction
oldbalanceDest	  Recipient balance before transaction
newbalanceDest	  Recipient balance after transaction
isFraud	✅       Target label (1 = fraud, 0 = legitimate)
balance_diff	    Engineered feature: difference in sender's balance
transfer_ratio	  Engineered feature: amount / oldbalanceOrg

🧪 Model Training
The training script (train_model.py) performs:
Data cleaning and preprocessing
Feature engineering
Class balancing using SMOTE
Training a Random Forest Classifier
Saving the model and scaler using joblib

To train via:
```bash
python train_model.py
```

🌐 Web Interface
Launch the app via:
bash
python app.py

Built with Flask
Upload or manually enter transaction values

Displays whether a transaction is legitimate or fraudulent

📌 Project Highlights
🚨 Handles severely imbalanced data using advanced techniques
💡 Feature engineering to improve fraud pattern detection
🖥️ End-to-end solution from data to web interface
🔍 Near-perfect AUC score (> 0.99)
Threshold tuning
Class weighting

Feature engineering is crucial for extracting patterns indicative of fraudulent behavior.
All sensitive fields like sender/receiver names (nameOrig, nameDest) are dropped for modeling.
