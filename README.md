🏥 ANN Healthcare Prediction System

A Deep Learning–based Healthcare Prediction Web Application built using Artificial Neural Networks (ANN) and deployed with Streamlit.

This project demonstrates an end-to-end Machine Learning pipeline including data preprocessing, encoding, feature scaling, model training, and deployment.

🔗 Live Application:
https://annhealthcare-9xegh9faeirddnnqkaeqmk.streamlit.app/

📌 Project Overview

The ANN Healthcare Prediction System predicts healthcare outcomes based on user-provided medical parameters.

This project showcases:

Data preprocessing

Label Encoding

One-Hot Encoding

Feature Scaling

ANN Model Training

Model Serialization

Deployment using Streamlit

It reflects practical implementation of Deep Learning concepts in a real-world-style predictive system.

🧠 Machine Learning Workflow

The application follows a structured prediction pipeline:

1️⃣ User Input

Users enter healthcare-related parameters via the Streamlit interface.

2️⃣ Label Encoding

Categorical variables are converted into numeric format using stored label encoder .pkl files.

3️⃣ One-Hot Encoding

Multi-category features are transformed using saved one-hot encoders.

4️⃣ Feature Scaling

Numerical inputs are standardized using a pre-trained scaler to maintain consistency with training data.

5️⃣ ANN Prediction

The processed input is passed into the trained Artificial Neural Network model for final prediction.

6️⃣ Result Display

Prediction results are shown instantly in the web application.

🏗️ Model Architecture

Algorithm: Artificial Neural Network (ANN)

Framework: TensorFlow / Keras

Task: Binary Classification

Hidden Layers: Dense Layers

Activation Function (Hidden): ReLU

Activation Function (Output): Sigmoid

Loss Function: Binary Crossentropy

Optimizer: Adam

The model is trained using backpropagation and saved as an .h5 file for deployment.

📂 Project Structure
ANN_Healthcare/
│
├── ANN_model/
│   └── model.h5
│
├── Label-encoders/
│   ├── encoder1.pkl
│   ├── encoder2.pkl
│   └── encoder3.pkl
│
├── One-hot-encoders/
│   ├── onehot1.pkl
│   └── onehot2.pkl
│
├── scaled_data/
│   └── scaler.pkl
│
├── app.py
├── requirements.txt
└── project.ipynb



Folder Explanation

ANN_model/ → Contains the trained neural network model

Label-encoders/ → Stores label encoder pickle files

One-hot-encoders/ → Stores one-hot encoder pickle files

scaled_data/ → Contains the saved scaler

app.py → Main Streamlit web application

project.ipynb → Model training and experimentation notebook

🛠️ Tech Stack

Python

TensorFlow / Keras

Scikit-Learn

Pandas

NumPy

Streamlit

Pickle

🚀 Features

✔ Interactive Web Interface
✔ Real-Time Healthcare Prediction
✔ Structured ML Preprocessing Pipeline
✔ Model & Encoder Persistence
✔ Clean Project Organization
✔ Deployed Live on Streamlit Cloud

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/ShivkumarNevhal/ANN_Healthcare.git
cd ANN_Healthcare

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Application
streamlit run app.py


The application will open in your default browser.

📊 Model Evaluation

Model performance evaluation was conducted during training in project.ipynb.

Metrics evaluated include:

Accuracy

Confusion Matrix

Precision

Recall

F1 Score

(You can update this section with your exact accuracy value if needed.)

🎯 Learning Outcomes

Through this project, I gained hands-on experience in:

Artificial Neural Networks

Feature Engineering

Data Encoding Techniques

Model Serialization

End-to-End ML Pipeline Design

Deployment of ML Models using Streamlit

Structuring ML projects professionally

🔮 Future Improvements

Add model comparison (ANN vs Random Forest vs XGBoost)

Add ROC Curve visualization

Improve UI/UX design

Convert to FastAPI backend architecture

Dockerize for production deployment

Add input validation and logging

👨‍💻 Author

Shivkumar Prabhakar Nevhal
MSc Computer Applications
Aspiring AI/ML Engineer

⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub!
