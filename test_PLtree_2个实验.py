'''
实验1:和CART树的比较
experiment 1 :
f(x) = 4sin(4*pi*t) - sign(t - 0.3) - sign(0.72 - t)


experiment 2: 二维分类问题
正类:x^2 + y^2<0.7
负类:else


'''





##---experiment2---#
import sys
import numpy as np
import xgboost as xgb
from linxgb import linxgb
from metrics import *
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from PL_Forest import RandomRotationPLForest

np.random.seed(0)

# Training set
train_n = 100
train_X = np.random.rand(train_n, 2)
c = ( np.square(train_X[:,0])+np.square(train_X[:,1]) < 0.7 )
train_Y = np.zeros(train_n)
train_Y[c] = 1
dtrain = xgb.DMatrix(train_X, label=train_Y.reshape(-1,1))


# plt.scatter(train_X[train_Y == 1,0],train_X[train_Y == 1,1],color = 'red')
# plt.scatter(train_X[train_Y == 0,0],train_X[train_Y == 0,1],color = 'blue')
# plt.show()

# Testing set
nx = ny = 100
X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
test_X = np.c_[ X.ravel(), Y.ravel() ]
dtest = xgb.DMatrix(test_X)

# plt.scatter(test_X[:,0],test_X[:,1],color = 'red')
# plt.show()


def plotsurf(ax, z):
    ax.axison = False
    Z = z.reshape(X.shape)
    red = [0.25,1,0.5]
    green = [1,0.3,0.3]
    ax.contourf(X, Y, Z, colors = [green,red], alpha = 0.2, levels = [0,0.5,1])
    ax.scatter(train_X[c,0],  train_X[c,1],  c = 'red', s = 20)
    ax.scatter(train_X[~c,0], train_X[~c,1], c = 'green',   s = 20)

# Common parameters, XGBoost 2 and LinXGBoost
num_trees = 1
learning_rate = 1
max_depth = 30
gamma = 5
subsample = 1.0
min_samples_leaf = 6

# XGBoost 1 (defaults with 50 trees) training
param = {'booster': 'gbtree', # gbtree, gblinear
         'eta': 0.1, # step size shrinkage
         'objective': 'binary:logistic' # binary:logistic, reg:linear
         }
num_round = 50 # the number of round to do boosting, the number of trees
bst1 = xgb.train(param, dtrain, num_round)

# XGBoost 2 (single tree) training
param = {'booster': 'gbtree', # gbtree, gblinear
         'eta': learning_rate, # step size shrinkage
         'gamma': gamma, # min. loss reduction to make another partition
         'min_child_weight': min_samples_leaf, # In regression, minimum number of instances required in a child node
         'max_depth': max_depth,  # maximum depth of a tree
         'lambda': 0.0, # L2 regularization term on weights, default 0
         'lambda_bias': 0.0, # L2 regularization term on bias, default 0
         'save_period': 0, # 0 means do not save any model except the final round model
         'nthread': 1,
         'subsample': subsample,
         'objective': 'binary:logistic' # binary:logistic, reg:linear
         # 'eval_metric': the evaluation metric
         }
num_round = num_trees # the number of round to do boosting, the number of trees
bst2 = xgb.train(param, dtrain, num_round)

# LinXGBoost
linbst = linxgb(objective="binary:logistic",
                n_estimators=num_trees,
                learning_rate=learning_rate,
                min_samples_leaf=min_samples_leaf,
                max_samples_linear_model=10000,
                max_depth=max_depth,
                subsample=subsample,
                lbda=0,
                gamma=gamma,
                prune=True,
                verbose=1)
linbst.fit(train_X, train_Y)

# CART
cart = DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf
                              )
cart.fit(train_X,train_Y)

#random Forest
rf = RandomForestClassifier(min_samples_leaf=min_samples_leaf,
                            max_depth = max_depth,
                            n_estimators=1000
                            )
rf.fit(train_X,train_Y)

# random rotation PL-Tree Forest

rrPLF = RandomRotationPLForest(objective= 'binary:logistic',
                               n_estimators=200,
                               max_depth = max_depth,
                               min_samples_leaf=min_samples_leaf,
                               max_samples_linear_model=10000,
                               subsample=subsample)
rrPLF.fit(train_X,train_Y)





# Plots
fig = plt.figure(figsize=(19,10), facecolor='white')
ax = fig.add_subplot(1, 6, 1, xticks=[], yticks=[])
z = bst1.predict(dtest)
z = np.array( [ row for row in z ] )
ax.set_title("XGBoost with 50 trees")
plotsurf(ax, z)
ax = fig.add_subplot(1, 6, 2, xticks=[], yticks=[])
z = bst2.predict(dtest)
z = np.array( [ row for row in z ] )
ax.set_title("XGBoost with 1 tree")
plotsurf(ax, z)
ax = fig.add_subplot(1, 6, 3, xticks=[], yticks=[])
z = linbst.predict(test_X)
ax.set_title("LinXGBoost with 1 tree")
plotsurf(ax, expit(z))
ax = fig.add_subplot(1, 6, 4, xticks=[], yticks=[])
z = cart.predict_proba(test_X)[:,1]
ax.set_title("CART")
plotsurf(ax, z)
ax = fig.add_subplot(1, 6, 5, xticks=[], yticks=[])
z = rf.predict_proba(test_X)[:,1]
ax.set_title("RandomForest")
plotsurf(ax, z)
ax = fig.add_subplot(1, 6, 6, xticks=[], yticks=[])
z = rrPLF.predict(test_X)
ax.set_title("random rotation PL-Tree Forest")
plotsurf(ax, expit(z))
plt.show()



##---experiment1---#
import numpy as np
import xgboost as xgb
from linxgb import linxgb
from metrics import *
from test_func import *
from test_plot import test_plot3a,test_subplot
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

reg_func = "heavysine"

# Noise level
np.random.seed(0)
s2 = 0.05

# Training set
train_X, train_Y = test_func(reg_func, n_samples=201, var=s2)
dtrain = xgb.DMatrix(train_X, label=train_Y.reshape(-1,1))

plt.scatter(train_X,train_Y)
plt.show()

# Testing set
test_X, test_Y = test_func(reg_func, n_samples=5000)
dtest = xgb.DMatrix(test_X)

# Common parameters, XGBoost 2 and LinXGBoost
num_trees = 1
learning_rate = 1
max_depth = 30
gamma = 3
subsample = 1.0
min_samples_leaf = 6

# XGBoost 1 (defaults with 50 trees) training
param = {'booster': 'gbtree', # gbtree, gblinear
         'eta': 0.1, # step size shrinkage
         #'gamma': 1, # min. loss reduction to make another partition
         #'min_child_weight': min_samples_leaf, # In regression, minimum number of instances required in a child node
         #'max_depth': max_depth,  # maximum depth of a tree
         #'lambda': 0.0, # L2 regularization term on weights, default 0
         #'lambda_bias': 0.0, # L2 regularization term on bias, default 0
         #'save_period': 0, # 0 means do not save any model except the final round model
         #'nthread': 1,
         #'subsample': subsample,
         'objective': 'reg:linear' # binary:logistic, reg:linear
         # 'eval_metric': the evaluation metric
         }
num_round = 50 # the number of round to do boosting, the number of trees
bst1 = xgb.train(param, dtrain, num_round)

# XGBoost 2 (single tree) training
param = {'booster': 'gbtree', # gbtree, gblinear
         'eta': learning_rate, # step size shrinkage
         'gamma': gamma, # min. loss reduction to make another partition
         'min_child_weight': min_samples_leaf, # In regression, minimum number of instances required in a child node
         'max_depth': max_depth,  # maximum depth of a tree
         'lambda': 0.0, # L2 regularization term on weights, default 0
         'lambda_bias': 0.0, # L2 regularization term on bias, default 0
         'save_period': 0, # 0 means do not save any model except the final round model
         'nthread': 1,
         'subsample': subsample,
         'objective': 'reg:linear' # binary:logistic, reg:linear
         # 'eval_metric': the evaluation metric
         }
num_round = num_trees # the number of round to do boosting, the number of trees
bst2 = xgb.train(param, dtrain, num_round)

# LinXGBoost training
linbst = linxgb(n_estimators=num_trees,
                learning_rate=learning_rate,
                min_samples_leaf=min_samples_leaf,
                max_samples_linear_model=10000,
                max_depth=max_depth,
                subsample=subsample,
                lbda=0,
                gamma=gamma,
                prune=True,
                verbose=1)
linbst.fit(train_X, train_Y)

# cart
cart = DecisionTreeRegressor(max_depth = max_depth,
                             min_samples_leaf= min_samples_leaf,
                             )
cart.fit(train_X,train_Y)

# random forest

RF = RandomForestRegressor(n_estimators=50,
                           min_samples_leaf= min_samples_leaf,
                           max_depth = max_depth)

RF.fit(train_X,train_Y)

# random rotation PL-Tree Forest
from PL_Forest import RandomRotationPLForest
rrPLF = RandomRotationPLForest(objective= "reg:linear",
                               n_estimators=50,
                               max_depth = max_depth,
                               min_samples_leaf=min_samples_leaf,
                               max_samples_linear_model=10000,
                               subsample=subsample,
                               random_ratation=True)
rrPLF.fit(train_X,train_Y)
# Make predictions
xgb1_pred_Y = bst1.predict(dtest)
xgb2_pred_Y = bst2.predict(dtest)
lin_pred_Y = linbst.predict(test_X)
cart_pred_Y = cart.predict(test_X)
RF_pred_Y = RF.predict(test_X)
rrPLF_pred_Y = rrPLF.predict(test_X)

# Print scores
print("NMSE: XGBoost1 {:12.5f}, XGBoost2 {:12.5f}, LinXGBoost {:12.5f},CART {:12.5f},RF {:12.5f},rrPLF {:12.5f}". \
       format(nmse(test_Y,xgb1_pred_Y),
              nmse(test_Y,xgb2_pred_Y),
              nmse(test_Y,lin_pred_Y) ,
              nmse(test_Y, cart_pred_Y),
              nmse(test_Y, RF_pred_Y),
              nmse(test_Y, rrPLF_pred_Y)))

# Plots
test_plot6a(reg_func, train_X, train_Y, test_X, test_Y,
            pred1_Y=xgb1_pred_Y, pred1_name="XGBoost",
            pred2_Y=xgb2_pred_Y, pred2_name="XGBoost",
            pred3_Y=lin_pred_Y,  pred3_name="LinXGBoost",
            pred4_Y=cart_pred_Y, pred4_name = 'CART',
            pred5_Y=RF_pred_Y,   pred5_name = 'RandomFforest',
            pred6_Y=rrPLF_pred_Y,   pred6_name = 'randomrotationPL-TreeForest',
            fontsize=25, savefig=True) # 36 for paper



