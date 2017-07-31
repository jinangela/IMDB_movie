import numpy as np
import pandas as pd
from itertools import chain
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer, FunctionTransformer

class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, *_):
        return X[self.columns]

    def fit(self, *_):
        return self

class OrdinalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def transform(self, X, *_):
        for i in range(len(self.columns)):
            X[self.columns[i]] = pd.factorize(X[self.columns[i]])[0]
        return X
    
    def fit(self, *_):
        return self

class UnknownImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def transform(self, X, *_):
        for i in range(len(self.columns)):
            if X[self.columns[i]].dtype.name == "category":
                X[self.columns[i]] = X[self.columns[i]].cat.add_categories("Unknown")
            X[self.columns[i]] = X[self.columns[i]].fillna("Unknown")
        return X
    
    def fit(self, *_):
        return self

class DtypeConverter(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def transform(self, X, *_):
        for i in range(len(self.columns)):
            X[self.columns[i]] = X[self.columns[i]].astype("category")
        return X
    
    def fit(self, *_):
        return self

class SmallCategoryCombiner(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def transform(self, X, *_):
        for i in range(len(self.columns)):
            tmp = pd.DataFrame(X[self.columns[i]].value_counts())
            X[self.columns[i]] = X[self.columns[i]].apply(lambda x: "Others" if x in tmp.loc[tmp[self.columns[i]] < 10].index.tolist() else x)
        return X
    
    def fit(self, *_):
        return self

class CategoricalTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.columns_ = None
        self.cat_columns_ = None
        self.non_cat_columns = None
        self.cat_map_ = None
        self.ordered_ = None
        self.dummy_columns_ = None
        self.transformed_columns_ = None

    def fit(self, X, y=None, *args, **kwargs):
        self.columns_ = X.columns
        self.cat_columns_ = X.select_dtypes(include=['category']).columns
        self.non_cat_columns_ = X.columns.drop(self.cat_columns_)

        self.cat_map_ = {col: X[col].cat.categories
                         for col in self.cat_columns_}
        self.ordered_ = {col: X[col].cat.ordered
                         for col in self.cat_columns_}

        self.dummy_columns_ = {col: ["_".join([col, v])
                                     for v in self.cat_map_[col]]
                               for col in self.cat_columns_}
        self.transformed_columns_ = pd.Index(
            self.non_cat_columns_.tolist() +
            list(chain.from_iterable(self.dummy_columns_[k]
                                     for k in self.cat_columns_))
        )

    def transform(self, X, y=None, *args, **kwargs):
        return (pd.get_dummies(X)
                  .reindex(columns=self.transformed_columns_)
                  .fillna(0))
    
    def fit_transform(self, X, y=None, *args, **kwargs):
        self.fit(X)
        return self.transform(X, y=None, *args, **kwargs)

    def inverse_transform(self, X):
        X = np.asarray(X)
        series = []
        non_cat_cols = (self.transformed_columns_
                            .get_indexer(self.non_cat_columns_))
        non_cat = pd.DataFrame(X[:, non_cat_cols],
                               columns=self.non_cat_columns_)
        for col, cat_cols in self.dummy_columns_.items():
            locs = self.transformed_columns_.get_indexer(cat_cols)
            codes = X[:, locs].argmax(1)
            cats = pd.Categorical.from_codes(codes, self.cat_map_[col],
                                             ordered=self.ordered_[col])
            series.append(pd.Series(cats, name=col))
        # concats sorts, we want the original order
        df = (pd.concat([non_cat] + series, axis=1)
                .reindex(columns=self.columns_))
        return df