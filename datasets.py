import numpy as np
import pandas as pd
import scipy.special
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone

def fetch_data_generator(*, data, semi_synth=False, simple_synth=False,
                         scale=1, true_f=None, max_depth=3):
    '''
    data: one of {'401k', 'criteo', 'welfare'}
    semi_synth: whether to impute outcomes from a synthetic model
    simple_synth: if outcome model should be simple based on the `true_f` function or fitted from data
    scale: how much noise to add for synthetic data generation
    true_f: a simple conditional expectation function for the outcome for semi synthetic data
    max_depth: if CEF for outcome is fitted from data, we will fit a random forest of this max_depth 
    '''

    if data == '401k':

        abtest = False
        file = "https://raw.githubusercontent.com/CausalAIBook/MetricsMLNotebooks/main/data/401k.csv"
        data = pd.read_csv(file)
        y = data['net_tfa'].values
        D = data['e401'].values
        X = data.drop(['e401', 'p401', 'a401', 'tw', 'tfa', 'net_tfa', 'tfa_he',
                        'hval', 'hmort', 'hequity',
                        'nifa', 'net_nifa', 'net_n401', 'ira',
                        'dum91', 'icat', 'ecat', 'zhat',
                        'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7',
                        'a1', 'a2', 'a3', 'a4', 'a5'], axis=1)
        
        mask = (X['inc'] > 0) & (X['inc'] >= np.percentile(X['inc'], 1))
        mask &= (X['inc'] <= np.percentile(X['inc'], 99))
        X, D, y = X[mask], D[mask], y[mask]
        inds = np.arange(X.shape[0])
        np.random.shuffle(inds)
        X, D, y = X.iloc[inds], D[inds], y[inds]

    elif data == 'criteo':
        
        abtest = True
        df = pd.read_csv('./criteo-uplift-v2.1.csv')
        y = df['visit'].values
        D = df['treatment'].values
        X = df.drop(['treatment', 'conversion', 'visit', 'exposure'], axis=1)
        
    elif data == 'welfare':

        abtest = True
        
        df = pd.read_csv('./welfarenolabel3.csv', na_values=-999)
        continuous = ['hrs1', 'income', 'rincome', 'age', 'polviews',
                    'educ', 'earnrs', 'sibs', 'childs', 'occ80', 'prestg80', 'indus80',
                    'res16', 'reg16', 'family16', 'parborn', 'maeduc', 'degree', 
                    'hompop', 'babies', 'preteen', 'teens', 'adults']
        categorical = ['partyid', 'wrkstat', 'wrkslf', 'marital', 'race', 'mobile16', 'sex', 'born']
        df = df[['y', 'w'] + continuous + categorical]
        df = df.dropna()
        df = df[~((df['polviews']>4) & (df['polviews'] < 5))]
        df = pd.get_dummies(df, columns=categorical, drop_first=True)
        
        y = df['y'].values
        D = df['w'].values
        X = df.drop(['y', 'w'], axis=1)

    # for semi-synthetic data generation
    if semi_synth:

        if simple_synth:

            def gen_epsilon(n):
                std = np.std(true_f(D, X))
                return np.random.normal(0, scale * std, size=n)

            def get_data():
                return X, D, true_f(D, X) + gen_epsilon(X.shape[0])

        else:

            true_model = RandomForestRegressor(min_samples_leaf=50, max_depth=max_depth)
            true_model.fit(np.hstack([D.reshape(-1, 1), X]), y)
            def true_f(D, X):
                return true_model.predict(np.hstack([D.reshape(-1, 1), X]))

            true_residuals = y - cross_val_predict(clone(true_model), X, y, cv=5)
            def gen_epsilon(n):
                return scale * np.random.choice(true_residuals, size=n)

            def get_data():
                return X, D, true_f(D, X) + gen_epsilon(X.shape[0])
    
    else:

        def get_data():
            return X, D, y
        
        def true_f(D, X):
            return np.zeros(D.shape[0])

    def true_cate(X):
        return true_f(np.ones(X.shape[0]), X) - true_f(np.zeros(X.shape[0]), X)

    return get_data, abtest, true_f, true_cate