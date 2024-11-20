import argparse
import pandas as pd
import numpy as np 
import psycopg2
import json
from pathlib import Path
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler
from queries import ICD_QUERY, ADMISSIONS_QUERY
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score

ICD_CM_9_TO_10_MAPPING_FILE = "/gpfs/data/bbj-lab/users/eddie/ethos-paper/ethos/data/icd_cm_9_to_10_mapping.csv.gz"


def connect_to_database(cdict:dict):
    """Connect to database
    @var cdict (str[str]) : credentials 
    
    @return cn : connection to schema 
    @return cur : cursor of the connection 
    """
    # Connect to local postgres version of mimic
    schema_name=cdict["schema_name"]
    # dbname=cdict["dbname"]
    try:
        cn = psycopg2.connect(
            host=cdict["host"],
            port=cdict["port"],
            user=cdict["user"],
            password=cdict["password"],
            dbname=cdict["dbname"],  # Replace with your actual database name
            options="-c client_encoding=UTF8"
        )
        print("Connected to the PostgreSQL server successfully!")

        print('Connected to postgres {}.{}.{}!'.format(int(cn.server_version/10000),
                                                    (cn.server_version - int(cn.server_version/10000)*10000)/100,
                                                    (cn.server_version - int(cn.server_version/100)*100)))
        cur = cur = cn.cursor()
        set_schema_path = f"SET search_path to {schema_name}; COMMIT;"
        cur.execute(set_schema_path)
        return cn, cur 
    
    except psycopg2.Error as error:
        print(error)
        return None, None

# function from ethos.tokenize.translation_base Class IcdMixin
def create_icd_9_to_10_translation():
    """Return a dict that maps icd codes from version 9 to version 10
    @return version_mapping (str[str])
    """
    version_mapping = pd.read_csv(ICD_CM_9_TO_10_MAPPING_FILE, dtype=str)
    version_mapping.drop_duplicates(subset="icd_9", inplace=True)
    version_mapping = version_mapping.groupby("icd_9").icd_10.apply(
        lambda values: min(values, key=len)
    )
    return version_mapping.to_dict()

def translate_icd_9_to_10(version_mapping:dict, left_digits:int|None,
                          df:pd.DataFrame, version_col:str='icd_version', code_col:str="icd_code") -> pd.DataFrame:
    """Map df icd code columns from version 9 to version 10
    @var version_mapping (str[str]) : dict that maps between icd code 9 and 10 
    @var left_digits (int) : icd code rolled up to certain left digits 
    @var df (pd.Dataframe) 
    @var version_col (str)
    @var code_col (str)

    @return df (pd.DataFrame) 
    """
    is_version_9 = df[version_col] == 9
    df[code_col] = df[code_col].astype(str).str.strip()
    df.loc[is_version_9, code_col] = df.loc[is_version_9, code_col].map(version_mapping).fillna(df[code_col])

    # Update icd_version to 10 for successfully mapped codes
    mask = is_version_9 & df[code_col].isin(version_mapping.values())
    df.loc[mask, version_col] = 10

    # roll up icd codes
    if left_digits is not None:
        df[code_col] = df[code_col].apply(lambda x: x[:left_digits] if len(x) > left_digits else x)

    return df

def load_query_data(path:str, query:str, cn):
    """Query data or load data if data are pre-saved. 
    @var path (str) 
    @var query (str)
    @var cn : connection to schema 

    @return df (pd.DataFrame)
    """
    path = Path(path).resolve()
    
    # if data pre-saved -> load 
    # else if connection to database is successful -> query data 
    # else : error
    if path.exists():
        df = pd.read_csv(path)
        return df
    elif cn is not None:
        df = pd.read_sql(query, cn)
        df.to_csv(path, compression="gzip", index=False)
        return df
    else:
        raise ValueError(f"{path} does not exist and fail to connect to PostgreSQL.")
    
def get_icd_code_dummies(data_path:str, col_path:str, df:pd.DataFrame, key_col="hadm_id", code_col="icd_code") -> pd.DataFrame:
    """Get dummies of icd codes for each key columns or load dummies if data are pre-saved. 
    @var data_path (str)
    @var col_path (str)
    @var df (pd.Dataframe) : df contains icd codes 
    @var key_col (str) : key column for groupby 
    @var code_col (str) 

    @return icd_dummies (pd.DataFrame) : key_col, icd_cols 
    """
    data_path = Path(data_path).resolve()
    col_path = Path(col_path).resolve()
    if data_path.exists():
        sM = sparse.load_npz(data_path)
        col_names_loaded = np.load(col_path, allow_pickle=True)
        icd_dummies = pd.DataFrame(data=sM.toarray(), columns=col_names_loaded)
    else: 
        icd_dummies = pd.get_dummies(df.set_index(key_col)[code_col]).groupby(key_col).max().reset_index()
        sM = sparse.csr_matrix(icd_dummies.values)
        sparse.save_npz(data_path, sM)
        column_names = icd_dummies.columns.tolist()
        np.save(col_path, column_names)

    return icd_dummies

def train_test_split(df:pd.DataFrame, time_col:str="admittime", key_col:str="subject_id", size:tuple=(0.7, 0.1, 0.2)):
    """Sort by time column and then split the dataset into train, validation, test datasets. 
    @var df (pd.DataFrame)
    @var time_col (str)
    @var key_col (str)
    @var size (tuple[float])

    @return 
    """
    assert len(size) == 3
    assert sum(size) == 1
    n = len(df)
    train_size = int(size[0]*n)
    valid_size = int(size[1]*n)

    # sort by time
    # Let's say size = (0.7, 0.1, 0.2) for train, validation, test datasets 
    # pick first 70% admissions as train 
    # rest being validation and test
    # if subjects in validation and test are in train dataset, filter out
    df = df.sort_values(by=time_col).reset_index(drop=True)
    train_dataset = df.iloc[:train_size, :]
    valid_dataset = df.iloc[train_size:(train_size+valid_size), :]
    test_dataset = df.iloc[(train_size+valid_size):, :]

    # remove patients in valid/test who are in training 
    subject_ids = train_dataset[key_col]

    mask = valid_dataset[key_col].isin(subject_ids)
    valid_dataset = valid_dataset[~mask]

    mask = test_dataset[key_col].isin(subject_ids)
    test_dataset = test_dataset[~mask]

    # shuffle
    train_dataset = train_dataset.sample(frac = 1)
    valid_dataset = valid_dataset.sample(frac = 1)
    test_dataset = test_dataset.sample(frac = 1)

    return train_dataset, valid_dataset, test_dataset

def sklearn_train_loop(model, train_X:pd.DataFrame, train_y:pd.DataFrame, test_X:pd.DataFrame, test_y:pd.DataFrame):
    print("-"*50)
    print(model)
    model.fit(train_X, train_y)

    for mode in ("train", "test"):
        print("*"*30)
        if mode == "train":
            X, y = train_X, train_y 
        else:
            X, y = test_X, test_y 
        
        preds = model.predict(X)

        # confusion matrix
        cm = confusion_matrix(y, preds, labels=model.classes_)
        print(f"{mode} confusion matrix:", cm)

        # recall score 
        print(f"{mode} recall:", recall_score(y, preds))

        # balanced accuracy
        balanced_accuracy = balanced_accuracy_score(y, preds)
        print("{} balanced accuracy: {:.2f}%".format(mode, balanced_accuracy*100))

        # RoC 
        print(f"{mode} RoC AUC score:", roc_auc_score(y, preds))

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--creds", default="creds.json", help="path to credentials of postgres database")
    parser.add_argument("--digits", type=int, default=4, help="ICD code max left digit")
    # parser.add_argument("--postgres", type=bool, default=True, help="whether to connect to postgres database")
    args = parser.parse_args()

    creds_path = Path(args.creds).resolve()
    with open(creds_path, "r") as f:
        cdict = json.load(f)
    cn, cur = connect_to_database(cdict=cdict)

    icd_code_digits = args.digits

    icd_df = load_query_data('data/icd.csv.gz', ICD_QUERY, cn)
    admissions_df = load_query_data('data/admissions.csv.gz', ADMISSIONS_QUERY, cn)
    
    # get dummies
    admissions_df = pd.get_dummies(admissions_df, columns=['gender'], drop_first=True)

    version_mapping = create_icd_9_to_10_translation()

    icd_df = translate_icd_9_to_10(version_mapping=version_mapping, left_digits=icd_code_digits, df=icd_df)

    icd_dummies = get_icd_code_dummies(
        data_path=f"data/icd_dummies_d{icd_code_digits}.npz", 
        col_path=f"data/icd_dummies_col_names_d{icd_code_digits}.npy", 
        df=icd_df
    )
    print(f"icd dataframe: shape is {icd_df.shape}. Dataframe peek:\n{icd_df.head(10)}")
    print(f"icd_dummies dataframe: shape is {icd_dummies.shape}. Dataframe peek:\n{icd_dummies.head(10)}")
    print(f"admissions dataframe: shape is {admissions_df.shape}. Dataframe peek:\n{admissions_df.head(10)}")

    # merge admissions and icd_dummies
    main_df = pd.merge(left=admissions_df, right=icd_dummies, on="hadm_id")
    print(f"main dataframe: shape is {main_df.shape}. Dataframe peek:\n{main_df.head(10)}")

    # split dataset 
    train_dataset, valid_dataset, test_dataset = train_test_split(df=main_df)

    # drop key columns and time columns
    train_dataset = train_dataset.drop(columns=['hadm_id', 'subject_id', 'admittime'])
    valid_dataset = valid_dataset.drop(columns=['hadm_id', 'subject_id', 'admittime']) 
    test_dataset = test_dataset.drop(columns=['hadm_id', 'subject_id', 'admittime'])

    # get X, y
    train_X, train_y = train_dataset.iloc[:, 1:], train_dataset.iloc[:, 0].values
    valid_X, valid_y = valid_dataset.iloc[:, 1:], valid_dataset.iloc[:, 0].values
    test_X, test_y = test_dataset.iloc[:, 1:], test_dataset.iloc[:, 0].values

    # normalization (mainly affect age column)
    scaler = MinMaxScaler()
    train_X = scaler.fit_transform(train_X)
    valid_X = scaler.transform(valid_X)
    test_X = scaler.transform(test_X)

    # models 
    sklearn_train_loop(LogisticRegression(class_weight='balanced', n_jobs=-1, max_iter=1_000_000), 
                train_X, train_y,
                test_X, test_y)
    
    sklearn_train_loop(RandomForestClassifier(max_depth=100, n_jobs=-1, class_weight='balanced'), 
            train_X, train_y,
            test_X, test_y)
    
    pos_weight = (len(train_y) - sum(train_y)) / sum(train_y)
    sklearn_train_loop(XGBClassifier(n_jobs=-1, scale_pos_weight=pos_weight), 
                train_X, train_y,
                test_X, test_y)








