import numpy as np

class RGBGrayscaleOLS:
    def __init__(self, n_samples=10, seed=None):
        self.n = n_samples
        self.seed = seed
        self.w_cie = np.array([0.299, 0.587, 0.114])

        self.X = None
        self.y_target = None
        self.XTX = None
        self.XTX_inv = None
        self.XTy = None
        self.w_estimated = None

        self.generate_data()
        self.compute_ols()

    def generate_data(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        
        self.X = np.random.randint(0, 256, size=(self.n, 3))
        self.y_target = self.X @ self.w_cie

    def compute_ols(self):
        self.XTX = self.X.T @ self.X
        self.XTX_inv = np.linalg.inv(self.XTX)
        self.XTy = self.X.T @ self.y_target
        self.w_estimated = self.XTX_inv @ self.XTy

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y_target

    def get_XTX(self):
        return self.XTX

    def get_XTX_inv(self):
        return self.XTX_inv

    def get_XTy(self):
        return self.XTy

    def get_weights(self):
        return self.w_estimated