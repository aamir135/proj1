import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from perceptron import Percep
from adaline import Ada
from Stochastic_Gradient_Descent import SGD
from sklearn import preprocessing
def main ():
    DataSetSelection=input("Please Enter the name of DataSet (iris or wine) ")
    ClassifierSelection = input ("Please Enter the name of Classification (perceptron or adaline or sgd ")
    df_original = pd.read_csv('D:\ML_project1\iris.csv', header = 0)
    df1_original = pd.read_csv('D:\ML_project1\wine.csv', header = 0)

    #print(df_original)
    #try to use lambda function
    #Y_df_modified = df_original.drop(['sepal_length', 'sepal_width', 'petal_length','petal_width'], axis=1)
    #X_df_modified = df_original.drop(['species'], axis=1)
    #print(X_df_modified)
    #print(Y_df_modified)
    X_data = np.random.rand(len(df_original)) < 0.71
    X_train = df_original[X_data]
    X_test =  df_original[~X_data]
    X_train1 = X_train.drop(['species'], axis=1)
    X_test1 = X_test.drop(['species'], axis=1)
    #Y_data = np.random.rand(len(Y_df_modified)) < 0.71
    Y_train = X_train.drop(['sepal_length', 'sepal_width', 'petal_length','petal_width'], axis=1)
    Y_test = X_test.drop(['sepal_length', 'sepal_width', 'petal_length','petal_width'], axis=1)
    #print(X_train)
    #print("------------------------------------------")
    #print(Y_train)
    #X = X_train.as_matrix(columns=None)
    #print(X_train)
    #YY_train = np.where(Y_train == 'setosa', 1, 0)
    #y_test_cross_check = np.where(Y_test == 'Iris-setosa', 1, 0)
    #print(len(df_original))
    XX_train1 = X_train1.iloc[:, :].values
    XX__train_normalized = preprocessing.normalize(XX_train1, norm='l2')
    XX_test1 = X_test1.iloc[:, :].values
    XX__test_normalized = preprocessing.normalize(XX_test1, norm='l2')
    yy_train1 = Y_train.iloc[:, :].values
    yy_train11 = np.where(yy_train1 == 'setosa', 1, -1)
    yy_test1 = Y_test.iloc[:, :].values
    yy_test11 = np.where(yy_test1 == 'setosa', 1, -1)
    #print(yy_test1)
    #print(len(X_train))
    # print("------------------------------------------")
    #print(len(X_test))
    # print("------------------------------------------")
    #print(len(Y_train))
    #print("------------------------------------------")
    #print(len(Y_test))

    #min_max_scaler_train = preprocessing.MinMaxScaler()
    #x_scaled_train = min_max_scaler_train.fit_transform([X_train])
    #min_max_scaler_test = preprocessing.MinMaxScaler()
    #x_scaled_train = min_max_scaler_train.fit_transform([X_test])
    #----------------------------------------------------------------
    X2_data = np.random.rand(len(df1_original)) < 0.71
    X2_train = df1_original[X_data]
    X2_test = df1_original[~X_data]
    X2_train1 = X2_train.drop(['Class'], axis=1)
    X2_test1 = X2_test.drop(['Class'], axis=1)
    # Y_data = np.random.rand(len(Y_df_modified)) < 0.71
    Y2_train = X2_train.drop(['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash','Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols','Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'], axis=1)
    Y2_test = X2_test.drop(['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash','Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols','Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'], axis=1)
    XX2_train1 = X2_train1.iloc[:, :].values
    XX2__train_normalized = preprocessing.normalize(XX2_train1, norm='l2')
    XX2_test1 = X2_test1.iloc[:, :].values
    XX2__test_normalized = preprocessing.normalize(XX2_test1, norm='l2')
    yy2_train1 = Y2_train.iloc[:, :].values
    yy2_train11 = np.where(yy2_train1 == 1, 1, -1)
    yy2_test1 = Y2_test.iloc[:, :].values
    yy2_test11 = np.where(yy2_test1 == 1, 1, -1)
    #print(yy2_test1)

    #----------------------------------------------------------------

    eta = 0.2
    n_iter = 50

    if (DataSetSelection == "iris"):
        if (ClassifierSelection == "perceptron"):
            per = Percep(eta=eta, n_iter=n_iter, random_state=1)
            per.fit(XX__train_normalized, yy_train11)
            count_per1 = 0
            print(per.fit(XX__train_normalized, yy_train11))
            prediction = per.predict(XX__test_normalized)
            for x in range(len(yy_test11)):
                if prediction[x] == yy_test11[x]:
                    count_per1 = count_per1 + 1
            pl.plot(range(1, len(per.errors) + 1), per.errors)
            accuracy_per = count_per1 / prediction
            error_per = 1 - count_per1 / prediction

            pl.xlabel('Attempts')
            pl.ylabel('Number of misclassification')
            pl.show()
        elif (ClassifierSelection == "adaline"):
            ad = Ada(eta=eta, n_iter=n_iter, random_state=1)
            ad.fit(XX__train_normalized, yy_train11)
            count_ad1 = 0
            print(ad.fit(XX__train_normalized, yy_train11))
            prediction = ad.predict(XX__test_normalized)
            for x in range(len(yy_test11)):
                if prediction[x] == yy_test11[x]:
                    count_ad1 = count_ad1 + 1
            pl.plot(range(1, len(ad.errors) + 1), ad.errors)
            accuracy_ad = count_ad1 / prediction
            error_ad = 1 - count_ad1 / prediction

            pl.xlabel('Attempts')
            pl.ylabel('Number of misclassification')
            pl.show()
        elif (ClassifierSelection == "sgd"):
            sg = SGD(eta=eta, n_iter=n_iter, random_state=1)
            sg.fit(XX__train_normalized, yy_train11)
            count_sg1 = 0
            print(sg.fit(XX__train_normalized, yy_train11))
            prediction = sg.predict(XX__test_normalized)
            for x in range(len(yy_test11)):
                if prediction[x] == yy_test11[x]:
                    count_sg1 = count_sg1 + 1
            pl.plot(range(1, len(sg.errors) + 1), sg.errors)
            accuracy_sg = count_sg1 / prediction
            error_sg = 1 - count_sg1 / prediction

            pl.xlabel('Attempts')
            pl.ylabel('Number of misclassification')
            pl.show()
        elif (ClassifierSelection == "onevsall"):
            print("onevsall")
        else:
            print("input right classifier")
    elif (DataSetSelection == "wine"):
        if (ClassifierSelection == "perceptron"):
            if (ClassifierSelection == "perceptron"):
                per1 = Percep(eta=eta, n_iter=n_iter, random_state=1)
                per1.fit(XX2__train_normalized, yy2_train11)
                count_per = 0
                print(per1.fit(XX2__train_normalized, yy2_train11))
                prediction = per1.predict(XX2__test_normalized)
                for x in range(len(yy2_test11)):
                    if prediction[x] == yy2_test11[x]:
                        count1 = count_per + 1
                pl.plot(range(1, len(per1.errors) + 1), per1.errors)
                accuracy_per1 =   count_per / prediction
                errorr_per1 = 1 - count_per / prediction

                pl.xlabel('Attempts')
                pl.ylabel('Number of misclassification')
                pl.show()
        elif (ClassifierSelection == "adaline"):
            ad1 = Ada(eta=eta, n_iter=n_iter, random_state=1)
            ad1.fit(XX2__train_normalized, yy2_train11)
            count_ad = 0
            print(ad1.fit(XX2__train_normalized, yy2_train11))
            prediction = ad1.predict(XX2__test_normalized)
            for x in range(len(yy2_test11)):
                if prediction[x] == yy2_test11[x]:
                    count_ad = count_ad + 1
            pl.plot(range(1, len(ad1.errors) + 1), ad1.errors)
            accuracy_ad1 = count_ad / prediction
            error_ad1 = 1 - count_ad / prediction

            pl.xlabel('Attempts')
            pl.ylabel('Number of misclassification')
            pl.show()
        elif (ClassifierSelection == "sgd"):
            sg1 = SGD(eta=eta, n_iter=n_iter, random_state=1)
            sg1.fit(XX2__train_normalized, yy2_train11)
            count_sg = 0
            print(sg1.fit(XX2__train_normalized, yy2_train11))
            prediction = sg1.predict(XX2__test_normalized)
            for x in range(len(yy2_test11)):
                if prediction[x] == yy2_test11[x]:
                    count_sg = count_sg + 1
            pl.plot(range(1, len(sg1.errors) + 1), sg1.errors)
            accuracy_sg1 = count_sg / prediction
            error_sg1 = 1 - count_sg / prediction

            pl.xlabel('Attempts')
            pl.ylabel('Number of misclassification')
            pl.show()
        elif (ClassifierSelection == "onevsall"):
            print("onevsall")
        else:
            print("input right classifier")
    else:
        print("please enter right parameters")

main()