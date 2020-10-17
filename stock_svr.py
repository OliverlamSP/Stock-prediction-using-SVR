import csv 
import numpy as np 
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm, grid_search
import numpy as np
import tqdm
'''
for find the best model grid serach
'''
def svc_param_selection(X, y, nfolds):
    Cs = np.arange(1,100,10)
    gammas = np.arange(0.1,1,0.1)
    epsilons =  np.arange(0,50,5)
    param_grid = {'C': Cs, 'gamma' : gammas,'epsilon': epsilons}
    grid_searchs = grid_search.GridSearchCV(SVR(kernel='rbf'), param_grid, cv=nfolds)
    grid_searchs.fit(X, y)
    grid_searchs.best_params_
    return grid_searchs.best_params_

def predict_prices(x,stock_number,no):
    df = pd.read_csv('{}.HK.csv'.format(stock_number))
    data_point = 30
    df1 = df.loc[x-data_point:x-1]
    df2 = df.loc[x-data_point:x] 
    df1['id'] = range(0,data_point)
    X = np.array(df1[['id']])
    df2['id'] = range(0,data_point+1) 
    # print(df2)
    X_1 = np.array(df2[['id']])
    y_close = df1['Close']
    # best_parameter_dict = svc_param_selection(X, y_close, 5)
    
    # svr_rbf = SVR(kernel = 'rbf', C=best_parameter_dict.get('C'),gamma = best_parameter_dict.get('gamma'),epsilon= best_parameter_dict.get('epsilon'))
    svr_rbf = SVR(kernel = 'rbf', C=1e3,gamma =0.1)
    svr_rbf.fit(X,y_close)
    y_close_real_data = list(df2['Close'])
    plt.scatter(X_1,y_close_real_data,color='black',label='Data')
    plt.scatter(X_1, svr_rbf.predict(X_1), color='red',label='RBF model')
    plt.plot(X_1, svr_rbf.predict(X_1)*1.02, color='red',label='RBF Model +2%')
    plt.plot(X_1, svr_rbf.predict(X_1)*0.98, color='red',label='RBF Model -2%')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('{0}{1}'.format(stock_number,no))
    plt.legend()
    plt.show()
    # return y_close_real_data[-1],svr_rbf.predict(X_1)[-1]
    return y_close_real_data[-1],svr_rbf.predict(X_1)[-1],svr_rbf.predict(X_1)[-1]*0.98
    

# real_price_list = []
# predicted_price_list = []
# print(predict_prices(247,'0293'))
for a in tqdm.tqdm(range(240,248)):
    print(predict_prices(a,'0293',a))
    # real_price,predicted_price = predict_prices(a,'0293')
    # real_price_list.append(real_price)
    # predicted_price_list.append(predicted_price)
# outside_low = 0
# outside_hi = 0
# count = 0
# total = 0
# for price in tqdm.tqdm(range(0,len(real_price_list))):
#     if (predicted_price_list[price] *1 > real_price_list[price]) and (predicted_price_list[price]*0.99 < real_price_list[price]):
#         count += 1
#         total += 1
#     elif predicted_price_list[price]*0.99 > real_price_list[price]:
#         outside_low += 1
#         total += 1

#     elif predicted_price_list[price]*1 < real_price_list[price]:
#         outside_hi += 1
#         total += 1
#     else:
#         total += 1
# print(count/total,outside_low/total,outside_hi/total)




