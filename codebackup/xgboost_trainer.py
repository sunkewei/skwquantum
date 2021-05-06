import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
from trainer.DataRetriever import PinzhongData
import os

def training_data_prepare(filepath, portion, recalculate, is_class):
    train_X = train_y = np.array([])
    file_list = os.listdir(filepath)
    for filename in file_list:
        if filename.endswith('.csv'):
            pinzhong = filename[:-4]
            print(pinzhong)
            if recalculate:
                data = PinzhongData(pinzhong, load_type='recalculate', data_dir='history/', feature_dir='features/')
            if not recalculate:
                data = PinzhongData(pinzhong, load_type='load_file', feature_dir='features/')
                print(data.X)
            X,y = data.down_sampling_features(portion, is_class=is_class)
            # print(data.evidence_label)
            # print(X)
            label_names = data.evidence_label
            if len(train_X) == 0:
                train_X = X
                train_y = y
                continue
            if len(X)==0:
                continue
            train_X = np.concatenate((train_X, X), axis=0)
            train_y = np.append(train_y, y)
            print(len(train_X), len(train_y))

    return train_X, train_y, label_names


if __name__ == '__main__':
    recalculate = True
    training_portion = 0.01
    if recalculate:
        X, y, feature_names = training_data_prepare('history', training_portion, recalculate=True, is_class=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234565)
    else:
        X_train,y_train, feature_names = training_data_prepare('history', training_portion, recalculate=False, is_class=True)
        X_test, y_test, _ = training_data_prepare('history', 0.1, recalculate=False, is_class=True)
        print(X_train)

    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 3,
        'gamma': 0.1,
        'max_depth': 6,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 3,
        'eta': 0.1,
        'seed': 1000,
        'nthread': 7,
    }
    plst = list(params.items())
    print(feature_names)
    print(X_train[0])
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=feature_names)
    num_rounds = 5000
    model = xgb.train(plst, dtrain, num_rounds)
    # model.feature_names = feature_names
    signature = 'allset_portion_05.20210404'

    model.save_model('model_repo/' + signature + '.' + str(training_portion) + '.model')
    model.dump_model('model_repo/' + signature + '.' + str(training_portion) + '.raw.txt')

    # 对测试集进行预测
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    ans = model.predict(dtest)

    # 计算准确率
    cnt1 = 0
    cnt2 = 0
    for i in range(len(y_test)):
        if ans[i] == y_test[i]:
            cnt1 += 1
        else:
            cnt2 += 1

    print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))

    # 显示重要特征
    plot_importance(model)
    plt.show()

    ypred = model.predict(xgb.DMatrix(X_test, feature_names=feature_names))
    cm = confusion_matrix(y_test,ypred)
    print(cm)
