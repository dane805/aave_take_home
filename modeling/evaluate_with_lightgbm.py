import lightgbm
import pandas as pd
from sklearn.model_selection import train_test_split

from evaluate import evaluate_model

def fit_lightgbm(
        parameters,
        X_train_train, X_train_test, y_train_train, y_train_test
):

    ## missing은 handle하는 게 default
    train_data = lightgbm.Dataset(X_train_train, label=y_train_train)
    test_data = lightgbm.Dataset(X_train_test, label=y_train_test)

    model = lightgbm.train(
        parameters,
        train_data,
        valid_sets=test_data,
        num_boost_round=5000,
        early_stopping_rounds=100,
        verbose_eval=20
    )


    return model

def _get_params():
    parameters = {
        'application': 'binary',
        'objective': 'binary',
        'metric': 'auc',
        'is_unbalance': 'true',
        'boosting': 'gbdt',
        'num_leaves': 31,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 20,
        'learning_rate': 0.05,
        'verbose': 1
    }

    ## tuned hp using bo
    bo_params = {'bagging_fraction': 0.9741567872981456,
                 'feature_fraction': 0.9,
                 'lambda_l1': 2.5291055896332906,
                 'lambda_l2': 0.0,
                 'max_depth': 9,
                 'min_child_weight': 36.880021622488194,
                 'min_split_gain': 0.1,
                 'num_leaves': 40}
    for k, v in bo_params.items():
        parameters[k] = v

    return parameters


def run(X_train, X_test, y_train, y_test, filter_feature_by_model_importance=False, verbose=20):
    """
    간단하게 모델링해서 테스트
    """

    X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(
        X_train, y_train, test_size=0.2
    )

    parameters = _get_params()

    model = fit_lightgbm(parameters, X_train_train, X_train_test, y_train_train, y_train_test)

    if filter_feature_by_model_importance:
        importance_df = pd.Series(model.feature_importance(), index=X_train_train.columns).sort_values(ascending=False)
        top_columns = importance_df.index[:filter_feature_by_model_importance]

        X_train_train = X_train_train[top_columns]
        X_train_test = X_train_test[top_columns]
        X_train = X_train[top_columns]
        X_test = X_test[top_columns]

        model = fit_lightgbm(parameters, X_train_train, X_train_test, y_train_train, y_train_test)

    y_pred_prob_train = model.predict(X_train)
    y_pred_prob_test = model.predict(X_test)

    auc_val = evaluate_model(y_train, y_test, y_pred_prob_train, y_pred_prob_test)

    return auc_val


