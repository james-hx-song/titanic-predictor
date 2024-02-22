import pandas as pd
import numpy as np
import numpy.typing as npt

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def get_feature_matrix(df: pd.DataFrame) -> npt.NDArray[np.float64]:
    '''
    Generate the Feature Matrix for naive soft-margin SVM implementation

    args:
    df: pd.DataFrame: the input DataFrame for the csv

    returns:

    '''

    # Extracting the features
    feature_matrix = np.zeros((df.shape[0], 6))

    # First feature: Ticket Class
    feature_matrix[:, 0] = df['Pclass']

    # Second: Sex
    # Male is 0, Female is 1
    feature_matrix[:, 1] = df['Sex'].map(lambda x : 1 if x == 'female' else 0)

    # Third: Age
    feature_matrix[:, 2] = df['Age']

    # Fourth: Siblings/Spouses Aboard
    feature_matrix[:, 3] = df['SibSp']

    # Fifth: Parents/Children Aboard
    feature_matrix[:, 4] = df['Parch']

    # Sixth: Embarked Location
    # C = 0, Q = 1, S = 2
    feature_matrix[:, 5] = df['Embarked'].map(lambda x : 0 if x == 'C' else (1 if x == 'Q' else 2))


    # Replace NaN values with the mean of the column (sometimes, the age of a passenger is not known)
    for i in range(6):
        feature_matrix[:, i] = np.nan_to_num(feature_matrix[:, i], nan=round(np.nanmean(feature_matrix[:, i])))

    return feature_matrix
    
    

    


