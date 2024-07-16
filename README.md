**Financial Risk Prediction App with Random Forest**

**Overview**

This project implements a financial risk prediction application using a Random Forest classifier. The application predicts whether a person is at financial risk based on various input features such as income, age, loan amount, credit score, employment details, and more.
![image](https://github.com/user-attachments/assets/7e3a2bbf-b259-4313-812e-7d8896363093)

_**MachineLearning.py**_

![image](https://github.com/user-attachments/assets/988e2d3f-342e-4506-9588-d7d86f9e61dd)

_**datatrain.csv**_

**Key Features**

- Data Preprocessing: The project includes data preprocessing steps such as handling categorical data and scaling numerical features using scikit-learn.
- Model Training: Utilizes a Random Forest classifier to train the predictive model on a dataset containing labeled financial data.
- Model Evaluation: Evaluates the model using metrics like accuracy, precision, and recall to assess its performance.
- Visualization: Includes visualizations of decision trees, ROC curves, feature importance, and AUC scores using matplotlib and PyQt.
  
**Usage**
  
- Data Input: Users can input various financial attributes through a GUI interface.
- Prediction: Based on user input, the app predicts whether the individual is at financial risk or not.
- Results Display: The predicted risk status and evaluation scores (accuracy, precision, recall) are displayed to the user.
  
**Technologies Used**

- Python, PyQt for GUI development
- scikit-learn for machine learning tasks (Random Forest classifier, metrics evaluation)
- matplotlib for data visualization (decision tree plots, ROC curves)
- Firebase Realtime Database for data storage and retrieval
  
**Future Improvements**

- Implementing additional machine learning algorithms for comparison.
- Enhancing the GUI for a more intuitive user experience.
- Integrating real-time data updates using Firebase for dynamic predictions.
