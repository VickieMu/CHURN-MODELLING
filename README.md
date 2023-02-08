# CHURN-MODELLING
Churn Modelling, it is a process of determining the likelihood of a customer leaving or churning that is used by companies. The requirements in Churn Modelling include; Python, Jupyter Notebook, Pandas, Numpy, Matplotlib and Scilit-learn.
The information acquired in Churn Modelling is used to improve customer retention and prevent churn. In order to perform this churning, we used a sample dataset to train and evaluate a churn model. This churning was performed using Python and Jupyter Notebook. The following are steps that were used to perform churn and in every step there are codes that were used to accomplish it.
Step one was Importing the necessary libraries.
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Logistic Regression.
    from sklearn.metrics import confusion_matrix, accuracy_score.
The second step was to load the dataset.
    dataset = pd.read_csv("churn_modelling (1).csv")
The third step was exploring the dataset.
    dataset.head()
    dataset.describe()
    dataset.info()
The fourth step was preprocessing the data.
    X = dataset.iloc[:,3:-1].values
    y = dataset.iloc[:,-1].values
    X_train, X_test, y_train, y_test =btrain_test_split(X,y,test_size =0.2,random_state = 0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
 The fifth step was training the model
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)
 The sixth step was making predictions and evaluating the model.
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Accuracy:",accuracy)
 This is how Churn modelling was able to be created using Python and Jupyter Notebook.
    
