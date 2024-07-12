import joblib
import numpy as np
from data_utils import load_data, preprocess_data, split_data, encode_labels
from vectorization_utils import tokenize_and_pad, tfidf_vectorize
from model_utils import build_lstm_model, build_cnn_model, train_and_evaluate_model, train_ml_model
from eda_utils import plot_label_distribution, plot_text_length_distribution, plot_word_cloud
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

def main():
    train_data = load_data('data/train.csv')
    test_data = load_data('data/test.csv')
    val_data = load_data('data/val.csv')

    # Preprocessing the datasets
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)
    val_data = preprocess_data(val_data)

    # Plot and save EDA graphs
    plot_label_distribution(train_data, save_path='plots/label_distribution.png')
    plot_text_length_distribution(train_data, save_path='plots/text_length_distribution.png')
    plot_word_cloud(train_data, save_path='plots/word_cloud.png')

    # Split the data
    X_train, y_train = split_data(train_data)
    X_test, y_test = split_data(test_data)
    X_val, y_val = split_data(val_data)

    # Encode labels
    y_train_one_hot, label_encoder = encode_labels(y_train)
    y_test_one_hot, _ = encode_labels(y_test)
    y_val_one_hot, _ = encode_labels(y_val)

    # Convert one-hot encoded labels to integer labels for ML models
    y_train_int = np.argmax(y_train_one_hot, axis=1)
    y_test_int = np.argmax(y_test_one_hot, axis=1)
    y_val_int = np.argmax(y_val_one_hot, axis=1)

    # TF-IDF Vectorization for Machine Learning Models
    X_train_tfidf, X_test_tfidf, X_val_tfidf = tfidf_vectorize(X_train, X_test, X_val)

    # Train and evaluate Logistic Regression Model
    log_reg_params = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['newton-cg', 'lbfgs', 'liblinear']
    }
    log_reg_model, log_reg_accuracy = train_ml_model(LogisticRegression(), log_reg_params, X_train_tfidf, y_train_int, X_test_tfidf, y_test_int, "Logistic Regression")

    # Train and evaluate Naive Bayes Model
    nb_params = {
        'alpha': [0.5, 1.0, 1.5]
    }
    nb_model, nb_accuracy = train_ml_model(MultinomialNB(), nb_params, X_train_tfidf, y_train_int, X_test_tfidf, y_test_int, "Naive Bayes")

    # Train and evaluate SVM Model
    svm_params = {
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf']
    }
    svm_model, svm_accuracy = train_ml_model(SVC(), svm_params, X_train_tfidf, y_train_int, X_test_tfidf, y_test_int, "Support Vector Machine")

    # Tokenize and pad sequences for LSTM and CNN
    max_len = 100
    X_train_pad, X_test_pad, X_val_pad, tokenizer = tokenize_and_pad(X_train, X_test, X_val, max_len)
    joblib.dump(tokenizer, 'models/tokenizer.pkl')

    # Define input dimensions
    input_dim = len(tokenizer) + 1

    # Train and evaluate deep learning models
    lstm_model = build_lstm_model(input_dim, max_len)
    lstm_model, lstm_accuracy = train_and_evaluate_model(lstm_model, X_train_pad, y_train_one_hot, X_val_pad, y_val_one_hot, X_test_pad, y_test_one_hot, "Bidirectional LSTM", epochs=5, batch_size=64)

    cnn_model = build_cnn_model(input_dim, max_len)
    cnn_model, cnn_accuracy = train_and_evaluate_model(cnn_model, X_train_pad, y_train_one_hot, X_val_pad, y_val_one_hot, X_test_pad, y_test_one_hot, "CNN", epochs=5, batch_size=64)

    # Compare all models and select the best one
    model_accuracies = {
        "Logistic Regression": log_reg_accuracy,
        "Naive Bayes": nb_accuracy,
        "Support Vector Machine": svm_accuracy,
        "Bidirectional LSTM": lstm_accuracy,
        "CNN": cnn_accuracy
    }

    best_model_name = max(model_accuracies, key=model_accuracies.get)
    best_model_accuracy = model_accuracies[best_model_name]
    print(f"The best model is {best_model_name} with an accuracy of {best_model_accuracy:.4f}")

    # Save the best model
    if best_model_name == "Logistic Regression":
        joblib.dump(log_reg_model, 'models/best_model_log_reg.pkl')
    elif best_model_name == "Naive Bayes":
        joblib.dump(nb_model, 'models/best_model_nb.pkl')   
    elif best_model_name == "Support Vector Machine":
        joblib.dump(svm_model, 'models/best_model_svm.pkl')
    elif best_model_name == "Bidirectional LSTM":
        lstm_model.save('models/best_model_lstm.keras')
    elif best_model_name == "CNN":
        cnn_model.save('models/best_model_cnn.keras')
    print(f"The best model {best_model_name} has been saved successfully.")

if __name__ == "__main__":
    main()
