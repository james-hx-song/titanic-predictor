import helper as hp
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

def test_feature_matrix(filename: str) -> None:
    '''
    Test and getthe feature matrix function
    '''
    df = hp.load_data(filename)
    feature_matrix = hp.get_feature_matrix(df)
    # assert feature_matrix
    # assert feature_matrix.shape[1] == 6
    # assert feature_matrix.shape[0] == df.shape[0]
    # assert feature_matrix.dtype == 'float64'
    print(feature_matrix)


def soft_SVM(filename: str) -> tuple[LinearSVC, float]:
    '''
    Train a soft-margin SVM model
    '''
    df = hp.load_data(filename)
    feature_matrix = hp.get_feature_matrix(df)
    labels = df['Survived']

    # model = LinearSVC(C=0.1, loss='hinge', penalty='l2', dual=True)
    model = LinearSVC(C=0.1, loss='squared_hinge', penalty='l1', dual=False)
    scores = cross_val_score(model, feature_matrix, labels, cv=5, scoring='accuracy')
    perf = scores.mean()
    print(f"Cross Validation Performance: {perf}")
    return model, perf

def get_t_data(filename: str) -> tuple:
    '''
    Get the feature matrix and labels
    '''
    test_feature_matrix(filename)
    df = hp.load_data(filename)
    feature_matrix = hp.get_feature_matrix(df)
    labels = df['Survived']
    return feature_matrix, labels

def main():
    filename = "train.csv"

    X_train, y_train = get_t_data(filename)
    SVM, cv = soft_SVM(filename)
    SVM.fit(X_train, y_train)


    # test_file = "test.csv"
    # X_test, y_test = get_t_data(test_file)
    # y_pred = SVM.predict(X_test)
    # print(f"Performance (accuracy) on test set: {accuracy_score(y_test, y_pred)}")

    






if __name__ == '__main__':
    main()

