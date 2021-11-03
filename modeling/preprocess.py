import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def get_train_from_app(app:pd.DataFrame) -> pd.DataFrame:
    df = app[app['type'] == 'train'].reset_index(drop=True)
    return df

def detect_cate_features(
        df_train:pd.DataFrame, df_test:pd.DataFrame,
        jaccard_sim_threshold:float=0.5, max_unique_cate_count:int=50
) -> pd.DataFrame:
    """
    1, 2, 3, .., k와 같은 형식의 칼럼은 더미 같다는 아이디어
    중간에 값이 빠질 수도 있으므로 threshold 적용 -> jaccard_sim_threshold
    범주 가지 수에도 제한 두기 -> max_unique_cate_count
    테스트에서 트레인에 없는 카테고리 잇으면 제외
    """
    feature_cols = [col for col in df_train.columns if 'col' in col]

    res = []
    for col in feature_cols:
        col_unique_values = df_train[col].dropna().unique().tolist()

        if (len(col_unique_values) <= 2):
            continue

        s1 = set(col_unique_values)
        s2 = set(range(len(s1)))
        s3 = set(range(1, len(s1) + 1))

        test_only_value_count = len(set(df_test[col].dropna().unique()) - set(df_train[col].dropna().unique()))
        res.append([
            col, len(col_unique_values), (len(s1 & s2) / len(s1 | s2)), (len(s1 & s3) / len(s1 | s3)),
            test_only_value_count
        ])

    col_is_cate = pd.DataFrame(res, columns=['col', 'nunique', 'jaccard_sim1', 'jaccard_sim2', "test_only"])

    col_cate = col_is_cate[(col_is_cate.jaccard_sim1 >= jaccard_sim_threshold) | ((col_is_cate.jaccard_sim2 >= jaccard_sim_threshold))]
    col_cate = col_cate[col_cate['nunique'] <= max_unique_cate_count]
    col_cate = col_cate[col_cate.test_only == 0]
    col_cate.sort_values('nunique')

    return col_cate

def _to_dummy(df, col):
    dummy_na = df[col].isna().sum() > 0
    temp_dummy = pd.get_dummies(df[col], drop_first=True, prefix=col, dummy_na=dummy_na)
    temp_dummy = temp_dummy.astype(np.int8)

    df = df.drop(col, axis=1)
    df = pd.concat([df, temp_dummy], axis=1)

    return df

def one_hot_encode_cate_features(df_train:pd.DataFrame, df_test:pd.DataFrame, col_cate:pd.DataFrame, etc_cate_threshold:int=10):
    """
    train에서 너무 가지수가 적은 예외 케이스는 제거 -> etc_cate_threshold
    *한 피쳐에서 그런 케이스가 2개 이상일 때만 예외로 몰아서 처리
    """

    df = pd.concat([
        df_train.assign(type='train'), df_test.assign(type='test')
    ])

    for col in col_cate.col:
        col_val_cnt = df_train[col].value_counts()

        minor_values = col_val_cnt[col_val_cnt <= etc_cate_threshold].index


        val_etc = df_train[col].min() - 1
        for minor_value in minor_values:
            df[col] = df[col].replace({minor_value: val_etc})
        if len(minor_values) > 1:
            print(f"{len(minor_values)}가지 카테고리 제거 -> {col_val_cnt[col_val_cnt <= 10].sum()}개의 관측치를 예외 케이스로 변경")

        df = _to_dummy(df, col)

    ## 방어로직
    assert (df['type'] == 'train').sum() == len(df_train)
    assert (df['type'] == 'test').sum() == len(df_test)

    df_train = df[df['type'] == 'train'].drop('type', axis=1)
    df_test = df[df['type'] == 'test'].drop('type', axis=1)
    return df_train, df_test

def remove_single_test_features(df_train, df_test):
    """테스트에서 범주가 1가지밖에 없는 케이스는 다 제거"""
    feature_cols = [col for col in df_train.columns if 'col' in col]
    unique_value_count_test = pd.Series({col: df_test[col].nunique() for col in feature_cols}).sort_values()

    unique_columns = unique_value_count_test[unique_value_count_test == 1].index

    df_train = df_train.drop(unique_columns, axis=1)
    df_test = df_test.drop(unique_columns, axis=1)
    return df_train, df_test

def remove_edge_cates(X_train, X_test, y_train, y_test, rare_threshold=0.001):
    """
    one hot 적용 후기 때문에, 값의 종류가 2개 뿐인 칼럼을 모두 대상으로(카테로 추정)
    test에서 값이 너무 희소한 경우 -> rare_threshold
    희소할 때 train에서 target 값이 unique한 관측치들은 룰베이스로 제외
    rule_predict의 key는 rule base로 제외할 칼럼
    value는 다시 dict로, minor_value를 가진 경우 target으로 예측
    해당 칼럼은 처리한 뒤 제거
    """
    feature_cols = [col for col in X_train.columns if col.startswith("col")]
    unique_value_count_test = pd.Series({col: X_train[col].nunique() for col in feature_cols}).sort_values()

    rule_predict = {}
    for col in unique_value_count_test[unique_value_count_test == 2].index:

        small_value_ratio = X_test[col].value_counts(normalize=True).min()
        if small_value_ratio < rare_threshold:

            col_minor_value = X_test[col].value_counts().idxmin()
            col_minor_target_dist = y_train.loc[X_train[col] == col_minor_value].value_counts()
            if len(col_minor_target_dist) == 1:
                rule_predict[col] = {
                    "minor_value": col_minor_value,
                    "target": col_minor_target_dist.index[0]
                }

    X_train = X_train.drop(rule_predict.keys(), axis=1)
    X_test = X_test.drop(rule_predict.keys(), axis=1)

    print(f"제거할 칼럼 수: {len(rule_predict)}")

    return X_train, X_test


def rebalance_smote(X_train:pd.DataFrame, y_train:pd.Series):
    """결측이 없어야 함"""
    sm = SMOTE(random_state=42)

    X_sm, y_sm = sm.fit_resample(X_train, y_train)

    return X_sm, y_sm


def remove_missing(df):
    """
    무지성으로 결측치 있는 칼럼 모두 제외
    """
    na_count = df.isna().sum()
    df = df[na_count[na_count==0].index]

    return df

def mark_missing(df_train, df_test):
    """
    결측치는 모두 존재하지 않았던 값으로 대체
    """
    for col in df_train.columns:
        value_to_impute = min(df_train[col].min(), df_test[col].min())
        df_train[col] = df_train[col].fillna(value_to_impute)
        df_test[col] = df_test[col].fillna(value_to_impute)

    return df_train, df_test
