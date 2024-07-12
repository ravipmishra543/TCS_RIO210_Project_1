# Project Overview

This project was developed during my Internship with TCSion, as part of my MCA degree program. The objective was to create a comprehensive Python package that automates the detection of various emotions from text paragraphs and predicts the overall emotion. This end-to-end solution integrates several components to ensure accurate and efficient emotion classification using advanced AI models.

## Acknowledgements

I am deeply grateful to my industry mentor, Dr. Himdweep Walia, whose guidance and unwavering support have been invaluable throughout my internship. Dr. Walia’s expertise and constructive feedback greatly enhanced my learning experience and contributed significantly to the successful completion of my project. I also extend my sincere thanks to TCSion and Amity Online University for offering me this invaluable opportunity. Their commitment to fostering educational and professional growth has been instrumental in my development. Furthermore, I would like to acknowledge the contributions of my colleagues and friends, whose insightful feedback and encouragement were crucial in navigating the challenges of the internship. Their support not only provided motivation but also enriched my understanding and approach to the tasks at hand.

## Project Description

The Python package provided in this repository is designed to handle the entire process of emotion detection from text data. It includes the following functionalities:
- **Data Preprocessing:** Preparing and cleaning the text data to ensure it's suitable for model training.
- **Feature Extraction:** Transforming text into numerical representations using techniques such as TF-IDF vectorization.
- **Model Training:** Training and evaluating multiple machine learning and deep learning models to identify the best performer.
- **Model Evaluation:** Assessing model performance based on metrics like accuracy, precision, recall, and F1 score.
- **Model Saving:** Automatically saving the best-performing model based on the evaluation metrics.

## Getting Started

To run this package, follow these steps:

1. **Install Dependencies:**
   Ensure that you have all the necessary Python libraries installed. You can install them using the `requirements.txt` file provided in the repository. Use the following command to install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Program:**
   Execute the main script to start the program. This script will handle the entire workflow, from data preprocessing to model evaluation. Run the following command:
   ```bash
   python main.py
   ```

3. **Monitor and Interact:**
   During execution, charts and diagrams may be generated to visualize various aspects of the data and model performance. Ensure you close these visualizations as they appear to allow the program to continue running smoothly.

4. **Final Model:**
   At the end of the program, the best-performing model will be saved based on its accuracy, precision, recall, and F1 score. This model will be available for future use and deployment.

## Utility Modules

The package includes several utility modules that streamline various tasks:

- **`data_utils.py`:** Contains functions for data loading, cleaning, and preparation.
- **`eda_utils.py`:** Provides functions for exploratory data analysis (EDA), including visualizations and data insights.
- **`vectorization_utils.py`:** Implements text vectorization techniques such as TF-IDF to convert text data into numerical formats.
- **`model_utils.py`:** Includes functions for building, training, and evaluating machine learning and deep learning models.
- **`main.py`:** The main script that orchestrates the entire workflow, integrating all utility functions and executing the core processes.

## Data Management

To prevent data leakage and ensure robust model evaluation, the original dataset has been divided into three distinct files:
- **`train.csv`:** Used for training the models.
- **`test.csv`:** Used for evaluating the model's performance.
- **`val.csv`:** Used for validating the model during training and fine-tuning.

By adhering to these practices, the project aims to deliver a reliable and effective solution for emotion detection from text.
