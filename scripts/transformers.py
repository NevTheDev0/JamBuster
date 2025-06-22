from sklearn.base import BaseEstimator, TransformerMixin

class FeatureDropperTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_cols=None, verbose=False):
        self.drop_cols = drop_cols if drop_cols is not None else [] #Allows dropping features that MIGHT not be defined 
        self.verbose = verbose #Added to avoid unnecessary errors

    def fit(self, X, y=None): 
        return self

    def transform(self, X):
        X_ = X.copy()
        safe_drop_cols = [col for col in self.drop_cols if col in X_.columns]
        if self.verbose:
            print(f"Dropping columns: {safe_drop_cols}")
        return X_.drop(columns=safe_drop_cols)

