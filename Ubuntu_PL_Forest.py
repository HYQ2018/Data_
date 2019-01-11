'''
random rotation matrix
'''
import numpy as np
from scipy import linalg as lg
def random_rotation_matrix_incl_flip(n):

    QR = np.matrix(np.random.normal(0,1,n**2).reshape(n,n))

    q, r = lg.qr(QR)

    M = np.dot(q,np.diag(np.sign(np.diag(r))))

    return M

def random_rotation_matrix(n):

    M = random_rotation_matrix_incl_flip(n)

    if lg.det(M) < 0:

        M[:,0] = -1*M[:,0]

    return M

def save_traindf(X,y,name):
	y = pd.DataFrame(y)
	X = pd.DataFrame(X)
	df = pd.concat([y,X],axis = 1)
	df.to_csv("/home/hyq/PL/"+name,index = False,header = None)

def save_predict(X,name):
	X = pd.DataFrame(X)
	df.to_csv("/home/hyq/PL/"+name,index = False,header = None)


class RandomRotationPLForest:

    def __init__(self,
                 objective = 'reg:linear',
                 n_estimators=5,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_samples_linear_model=10000,
                 subsample = 1.0,
                 verbose=0,
                 random_ratation = True,
                 prune = False,
                 bootstrap = False,
                 num_classes = 1):

        self.loss_func = objective
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_samples_linear_model = max_samples_linear_model
        self.subsample = subsample
        self.verbose = verbose
        self.random_ratation = random_ratation
        self.trees = []
        self.prune = prune
        self.rotation = []
        self.bootstrap = bootstrap
        if objective == "l2":
        	self.eval_metric = "rmse"
        else:
        	self.eval_metric = "auc"

        if objective == "multi-logistic":
        	self.classes = num_classes
        else:
        	self.classes = 1
       	e

    def fit(self,X,y):

        M = np.diag([1 for i in range(X.shape[1])])

        for i in range(self.n_estimators):

            if self.bootstrap:

                index = np.arange(0,X.shape[0],1)
                index_ = np.random.choice(index,size= int(X.shape[0]*self.subsample),
                                          replace=False)
                X_train = X[index_,:]
                y_train = y[index_]

            else:

                X_train = X
                y_train = y

            if self.random_ratation:

                M = random_rotation_matrix(X.shape[1])

            X_train = np.dot(X_train,M)

            self.rotation.append(M)

            	params = {"num_leaves":2**self.max_depth-1,
            			"num_trees":1,
            			"min_sum_hessians":0,
            			"lambda":0,
            			"objective":self.loss_func,
            			"learning_rate":0,
            			"eval_metric":self.eval_metric,
            			"num_classes":self.classes,
            			"num_threads":1,
            			"num_bins":255,
            			"min_gain":0,
            			"max_var":X_train.shape[1],
            			"grow_by":"leaf",
            			"leaf_type":'linear',
            			"verbose":1,
            			"sparse_ratio":0.0,
						"boosting_type":"gbdt",
						 "goss_alpha":0.0, 
						 "goss_beta":0.0}
            			
				save_df(X_train,y_train)
				train_data = gbdtpl.DataMat("train", params, 0, -1, "/home/hyq/train.csv")
    			test_data = gbdtpl.DataMat("test", params, 0, -1, "/home/hyq/train.csv", train_data)
    			booster = gbdtpl.Booster(params, train_data, test_data) 
    			booster.Train()
                self.trees.append(booster)

    def predict(self, X):
        """Make predictions.
        """
        n = X.shape[0]
        y = np.zeros(n, dtype=float)
        for t in range(len(self.trees) ):
            X_ = np.dot(X,self.rotation[t])
            save_predict(X_,'predict.csv')
            X_ = gbdtpl.DataMat("predict", params, -1, -1, "/home/hyq/PL/predict.csv", train_data)
            y += self.trees[t].Predict(X_)
        y /= len(self.trees)
        return y