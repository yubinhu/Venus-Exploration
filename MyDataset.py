import numpy as np
import matplotlib.pyplot as plt

class MyDataset():
    def __init__(self, train_X, valid_X, train_y, valid_y, X_test = None):
        self.X_train = train_X
        self.y_train = train_y
        self.X_valid = valid_X
        self.y_valid = valid_y
        self.X_test = X_test
        
        self.normalized = False
        
    def __str__(self):
        return str((self.train_X.shape, self.train_y.shape, self.valid_X.shape, self.valid_y.shape))

    def normalize(self):
        if self.normalized:
            print("already normalized!")
            return
        
        std_list = np.std(self.X_train, axis=0)
        std_list = np.where(std_list == 0, 1, std_list) # prevent zero division error
        avg_list = np.mean(self.X_train, axis=0)
        self.X_train = (self.X_train - avg_list)/std_list
        if not self.X_valid is None: self.X_valid = (self.X_valid - avg_list)/std_list
        if not self.X_test is None: self.X_test = (self.X_test - avg_list)/std_list
            
        self.std_list = std_list
        self.avg_list = avg_list
        print("dataset normalized!")
        self.normalized = True
        return
    
    def normalize_vec(self, X):
        if not self.normalized:
            print("Dataset never normalized")
        return (X - self.avg_list)/self.std_list
    
    def add_fict(self):
        add_ones = lambda X: np.hstack((X,np.ones((X.shape[0],1))))
        self.X_train = add_ones(self.X_train)
        self.X_valid = add_ones(self.X_valid)
        self.X_test = add_ones(self.X_test)
        print("ficticious dimension added!")
        
    def print_dim(self):
        print("X_train has shape:",self.X_train.shape)
        print("y_train has shape:",self.y_train.shape)
        print("X_valid has shape:",self.X_valid.shape)
        print("y_valid has shape:",self.y_valid.shape)
        if self.X_test: print("X_test has shape:",self.X_test.shape)

def plot_acc(xl, yl, zl, filename, title):
    plt.plot(xl, yl, label="validation")
    plt.plot(xl, zl, label="training")
    plt.title(title)
    plt.xlabel("number of training data points")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(filename)
    plt.show()