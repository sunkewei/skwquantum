import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
import numpy as np
from sklearn.metrics import confusion_matrix
from trainer.DataRetriever import PinzhongData
import os
from matplotlib import pyplot


def training_data_prepare(filepath, portion, recalculate=True):
    label_names = []
    train_X = train_y = np.array([])
    file_list = os.listdir(filepath)
    for filename in file_list:
        if filename.endswith('.csv'):
            pinzhong = filename[:-4]
            print(pinzhong)
            if recalculate:
                data = PinzhongData(pinzhong)
                data.recalculate(training_data_dir='history/', feature_dir='features/')
            if not recalculate:
                data = PinzhongData(pinzhong)
                data.load_features_from_file(feature_dir='features/')
                print(data.X)
            down_sample_x, down_sample_y = data.down_sampling_features(portion=portion)
            # print(data.evidence_label)
            label_names = data.feature_names[:-1]
            if len(train_X) == 0:
                train_X = down_sample_x
                train_y = down_sample_y
                continue
            if len(down_sample_x) == 0:
                continue
            train_X = np.concatenate((train_X, down_sample_x), axis=0)
            train_y = np.append(train_y, down_sample_y)
            print(len(train_X), len(train_y))
        # break

    return train_X, train_y, label_names


def tuning_parameters(train_X, train_y):
    local_model = xgb.XGBClassifier(objective='multi:softmax',
                                    n_estimators= 11,
                                    booster='gbtree',
                                    use_label_encoder=False,
                                    max_depth=7,
                                    learning_rate=0.5,
                                    gamma=0.1,
                                    subsample=1,
                                    colsample_bytree=1,
                                    # min_child_weight=3,
                                    eval_metric='auc',
                                    tree_method='gpu_hist',
                                    gpu_id=0
                                    )
    # grid search
    # n_estimators = range(3, 1100, 200)
    # param_grid = dict(n_estimators=n_estimators)

    # max_depth = range(1, 17, 2)
    # print(max_depth)
    # param_grid = dict(max_depth=max_depth)
    learning_rate = (np.array(range(40, 60, 2))/100).tolist()
    param_grid = dict(learning_rate=learning_rate)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    grid_search = GridSearchCV(local_model, param_grid, scoring="accuracy", n_jobs=-1, cv=kfold, verbose=1)
    grid_result = grid_search.fit(train_X, train_y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    # plot
    pyplot.errorbar(learning_rate, means, yerr=stds)
    pyplot.title("XGBoost n_estimator vs accuracy")
    pyplot.xlabel('max_depth')
    pyplot.ylabel('Log Loss')
    pyplot.show()


def calc_precision(test, pred):
    cnt1 = 0
    cnt2 = 0
    for i in range(len(test)):
        if pred[i] == test[i]:
            cnt1 += 1
        else:
            cnt2 += 1
    return 100 * cnt1 / (cnt1 + cnt2)


if __name__ == '__main__':
    recalculate_flag = False
    training_portion = 0.6
    if recalculate_flag:
        X, y, feature_names = training_data_prepare('history', portion=training_portion, recalculate=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234565)

    else:
        X_train,y_train, feature_names = training_data_prepare('history', portion=training_portion, recalculate=False)
        X_test, y_test, _ = training_data_prepare('history', portion=1, recalculate=False)
        print(X_train)

    params = {
        'n_estimator': 11,
        'booster': 'gbtree',
        'objective': 'multi:softprob',
        'num_class': 3,
        'gamma': 0.1,
        'max_depth': 7,
        'lambda': 2,
        'subsample': 1,
        'colsample_bytree': 1,
        'min_child_weight': 3,
        'eta': 0.5,
        'seed': 1000,
        'nthread': 7,
        'eval_metric': 'auc',
        'tree_method': 'gpu_hist',
        'gpu_id': 0
    }
    # tuning_parameters(X_train, y_train)
    # exit(2223)
    plst = list(params.items())
    print(feature_names)
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=feature_names)
    num_rounds = 5000
    # model = xgb.train(plst, dtrain, num_rounds)

    model = xgb.XGBClassifier(objective='multi:softprob',
                              n_estimators=11,
                              booster='gbtree',
                              use_label_encoder=False,
                              max_depth=7,
                              learning_rate=0.5,
                              gamma=0.1,
                              subsample=1,
                              colsample_bytree=1,
                              # min_child_weight=3,
                              eval_metric='auc',
                              tree_method='gpu_hist',
                              gpu_id=0
                              )
    # print(y_train)
    model.fit(X_train, y_train)

    # model.feature_names = feature_names
    signature = 'allset_portion_05.20210404'

    model.save_model('model_repo/' + signature + '.' + str(training_portion) + '.model')
    # model.dump_model('model_repo/' + signature + '.' + str(training_portion) + '.raw.txt')

    # 对测试集进行预测
    # dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    ans = model.predict_proba(X_test)
    # print(ans)
    ans =np.array(np.array(ans).argmax(axis=1).T.tolist())
    # print(ans)
    print("Accuracy: %.2f %% " % (calc_precision(y_test, ans)))
    ypred_raw = model.predict_proba(X_test) # xgb.DMatrix(X_test, feature_names=feature_names))
    # print(ypred_raw)
    ypred = np.array(ypred_raw).argmax(axis=1).T
    y_test_refined = []
    y_predict_refined = []
    for idx, item in enumerate(ypred_raw):
        if abs(item[ypred[idx]]) > 0.85:
            y_predict_refined.append(ypred[idx])
            y_test_refined.append(y_test[idx])

    # cm = confusion_matrix(y_test,ypred)
    cm = confusion_matrix(y_test_refined, y_predict_refined)
    print(cm)
    print("Accuracy: %.2f %% " % (calc_precision(y_test_refined, y_predict_refined)))
    plot_importance(model)
    plt.show()

