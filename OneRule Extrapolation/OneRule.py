# import pandas as pd
# import numpy as np

# strNK = input()
# list_to_int = lambda x: (int(i) for i in x)
# N, K = list_to_int(str(strNK).split())
# features_name = input().split()
# features_name.append('class')
# train_list = []
# for i in range(N):
#     row = input().split()
#     train_list.append(row) 

# M = int(input())
# test_list = []
# for i in range(M):
#     row = input().split()
#     test_list.append(row)

# df_train = pd.DataFrame(train_list, columns=features_name)
# df_test = pd.DataFrame(test_list, columns=features_name[:K])

# def predict(_frame, _model):
#     col_name = _model.index.name
#     return _model.loc[_frame[col_name], 'ans'].values

# def accuracy(_predict, _answer):
#     count_true_ans = np.count_nonzero(_predict==_answer)
#     accur = count_true_ans / len(_predict)
#     return accur

# def fit_model(_train, _answer):
#     best_model = pd.DataFrame()
#     max_accur = 0
#     features = _train.columns.values
#     for feature in features:
#         model = pd.DataFrame([_train[feature], _answer]).transpose().groupby(feature).sum()
#         model['1'] = model.apply(lambda x: x['class'].count('1'), axis=1)
#         model['0'] = model.apply(lambda x: x['class'].count('0'), axis=1)
#         model['ans'] = model.apply(lambda x: 1 if x['1'] >= x['0'] else 0, axis=1)
#         #model.drop(['class', '1', '0'], axis=1, inplace=True)
#         pred = predict(_train, model)
#         accur = accuracy(pred, [int(item) for item in _answer.values])
#         if accur > max_accur:
#             max_accur = accur
#             best_model = model
#     return best_model

# model = fit_model(df_train.iloc[:,:-1], df_train.iloc[:,-1])
# pred = predict(df_test, model)
# for i in pred: print(i)

# import numpy as np
# import pandas as pd
# import numba

# @numba.njit(fastmath=True)
# def fit(_x, _y):
#     best_model = None
#     max_accur = 0
#     # for feature in _x.columns.values:
#     #     pass
#     print(_x)
#         # model = pd.DataFrame([_x[feature], _y['Class']]).transpose().groupby(feature).mean()
#         # model['Answer'] = model.apply(lambda x: 1 if x['Class'] >= 0.5 else 0, axis=1)
#         # pred = predict(_x, model)
#         # accur = accuracy(pred, [int(item) for item in _y.values])
#         # if accur > max_accur:
#         #     max_accur = accur
#         #     best_model = model
               
#     #for i in predict(test_x, best_model): print(i) 

# @numba.njit(fastmath=True)
# def accuracy(_predict, _answer):
#     count_true_ans = np.count_nonzero(_predict==_answer)
#     accur = count_true_ans / len(_predict)
#     return accur

# @numba.njit(fastmath=True)
# def predict(_x, _model):
#     col_name = _model.index.name
#     return _model.loc[_x[col_name], 'Answer'].values
    
# N, K = (int(item) for item in input().split())
# features_name = np.array(input().split())
# train_list = np.array([input().split() for i in range(N)])
# M = int(input())
# test_list = np.array([input().split() for i in range(M)])

# # train_x = pd.DataFrame(data=train_list[:,:-1], columns=features_name)
# # train_y = pd.DataFrame(data=[int(i) for i in train_list[:,-1]], columns=['Class'])
# # test_x = pd.DataFrame(data=test_list, columns=features_name)

# fit(train_list[:,:-1], train_list[:,-1])


import numpy as np

N, K = (int(item) for item in input().split())
features_name = np.array(input().split())
train_list = np.array([input().split() for i in range(N)])
M = int(input())
test_list = np.array([input().split() for i in range(M)])

train_x = train_list.transpose()
train_y = np.array([int(i) for i in train_list[:,-1]], dtype=int)
test_x = test_list.transpose()


def predict(_x, _model):
    pred = []
    for i in _x[_model[-1][0]]:
        ans = np.array([i[0] for i in _model[:-1]])
        marks = np.array([i[1] for i in _model[:-1]])
        index = np.where(ans == i)[0][0]
        pred.append(marks[index])
    return np.array(pred)

     
def accuracy(_predict, _answer):    
    count_true_ans = np.count_nonzero(_predict==_answer)
    accur = count_true_ans / len(_predict)
    return accur


def fit(_x, _y):
    best_model = None
    max_accur = 0
    for number in range(K):
        
        model = []
        
        unique_items = np.unique(_x[number])
        test_arr = np.array(list(zip(_x[number], _y)))
        uniq_test, counts = np.unique(test_arr, axis=0, return_counts=True)

        for item in unique_items:
            index = np.where(uniq_test == item)[0]
            ans = counts[index]
            mark = uniq_test[index][0][1]

            if len(ans) == 1 and mark == '1':
                model.append([uniq_test[index][0][0], 1])
                #print(uniq_test[index][0][0],  ans[0], 0)
            elif len(ans) == 1 and mark == '0':
                model.append([uniq_test[index][0][0], 0])
                #print(uniq_test[index][0][0], 0, ans[0])
            elif len(ans) == 2 and mark == '1':
                model.append([uniq_test[index][0][0], 0 if ans[0] >= ans[1] else 1])
                #print(uniq_test[index][0][0], ans[0], ans[1])
            else:
                model.append([uniq_test[index][0][0], 1 if ans[1] >= ans[0] else 0])
                #print(uniq_test[index][0][0], ans[1], ans[0])
        model.append([number])
        model = np.array(model)
        pred = predict(_x, model)
        accur = accuracy(pred, _y)
        if accur > max_accur:
            max_accur = accur
            best_model = model

    for i in predict(test_x, best_model): print(i)
    
        
fit(train_x, train_y)



