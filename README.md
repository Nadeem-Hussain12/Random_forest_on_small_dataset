# Random_forest_on_small_dataset
The provided Python program is focused on building and evaluating a machine learning model, specifically a Random Forest Classifier, to predict survival outcomes for passengers on the Titanic. Here's a 10-line description of the program:

It begins by importing necessary libraries such as NumPy, Pandas, and scikit-learn modules for data manipulation, modeling, and evaluation.

The Titanic dataset is read from a CSV file located at "F:/Nust/titanic_clean.csv" and stored in a Pandas DataFrame called df.

Data preprocessing is performed. Categorical variables such as 'Pclass', 'Sex', 'Embarked', 'Title', 'GrpSize', 'FareCat', and 'AgeCat' are converted into one-hot encoded vectors using Pandas' get_dummies function. This is done to make the data suitable for machine learning.

The target variable 'Survived' and any unnecessary features like 'PassengerId' are separated from the dataset. The feature matrix X contains all the predictor variables, while the target vector Y contains the survival labels.

The dataset is split into training and testing sets using scikit-learn's train_test_split function. It allocates 70% of the data for training and 30% for testing, ensuring reproducibility with a random seed of 100 and shuffling the data.

A Random Forest Classifier (clf_rf) is instantiated as the machine learning model.

The Random Forest Classifier is trained on the training data (xtrain and ytrain) using the fit method.

The model is used to make predictions on the test dataset (xtest) using predict and predict_proba methods to get both class predictions and class probabilities.

The program calculates the accuracy of the model's predictions by comparing them to the actual survival outcomes in the test set using accuracy_score from scikit-learn's metrics.

Finally, the program prints out the accuracy score, which represents the proportion of correctly predicted survival outcomes by the Random Forest Classifier on the test dataset. This score is a measure of the model's performance in classifying passengers as survivors or non-survivors based on the given features.
