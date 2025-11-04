import pandas as pd
import numpy as np

class MyLineReg():
    def __init__(self, n_iter: int = 100, learning_rate: float = 0.1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False) -> None:
        num_feat = len(X.columns)    # number of features
        num_calc = len(X)            # number of observations
        X.insert(0, 'w0', 1)
        self.weights = np.ones(num_feat+1)
        for i in range(self.n_iter):
            y_pred = X.dot(self.weights)
            loss_val = 1/num_calc*sum((y_pred-y)**2)
            MSE_grad = 2/num_calc*X.T.dot(y_pred-y)
            self.weights -= self.learning_rate*MSE_grad
            if verbose and i % verbose == 0:
                iter_info = i
                if i == 0:
                    iter_info = "start"
                print(f"{iter_info} | loss: {loss_val}")
    
    def predict(self, X: pd.DataFrame,):
        '''Predict results by linear regression model'''
        X.insert(0, 'w0', 1)
        y_pred = X @ self.weights
        return sum(y_pred)
        
        
    def get_coef(self):
        return self.weights[1:]
        
    def __str__(self) -> str:
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
    
    
if __name__ == "__main__":
    model = MyLineReg()
    X_test = pd.DataFrame({'feature1': [0.1, 0.2, 0.3, 0.5, 0.6], 'feature2': [0.4, 0.5, 0.6, 0.2, 0.4], 'feature3': [0.4, 0.5, 0.6, 0.2, 0.4]})
    y_test = pd.Series([0.1, 0.2, 0.3, 0.5, 0.4])
    # print(X)
    model.fit(X_test, y_test, 5)