import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import cross_validate as surprise_cv
from surprise.model_selection import train_test_split as surprise_split

from tqdm.auto import tqdm
from data_handler.data_handler import *

from config import svd_params as params, general_params
seed = general_params['seed']
ds_name = params['ds_name'] 
n_factors = params['n_factors'] 
lr = params['lr'] 
reg = params['reg'] 


def integrate_wishlist(ratings, tbr):
    typ = ds_name
    row_means = ratings.groupby('sid')['rating'].mean()
    column_means = ratings.groupby('pid')['rating'].mean()
    global_mean = ratings['rating'].mean()
    tbr_unique = tbr.loc[
        ~tbr.set_index(['sid','pid']).index.isin(
            ratings.set_index(['sid','pid']).index
        )
    ].copy()
    if typ == 'row':
      tbr_unique['rating'] = tbr_unique['sid'].map(row_means).fillna(global_mean)
    elif typ == 'column':
      tbr_unique['rating'] = tbr_unique['pid'].map(column_means).fillna(global_mean)
    elif typ == 'global':
      tbr_unique['rating'] = global_mean
    elif typ == 'const':
      tbr_unique['rating'] = 3.0
    return pd.concat([ratings, tbr_unique[['sid','pid','rating']]], ignore_index=True)

def cv(ratings, n_splits):
    r = Reader(rating_scale=(ratings.rating.min(), ratings.rating.max()))
    data = Dataset.load_from_df(ratings[['sid','pid','rating']], r)
    algo = SVD(n_factors=n_factors, lr_all=lr, reg_all=reg)
    res = surprise_cv(algo, data, measures=['RMSE'], cv=n_splits, verbose=True, return_train_measures=True)
    return np.mean(res['train_rmse']), np.std(res['train_rmse']), np.mean(res['test_rmse']), np.std(res['test_rmse'])

def train(ratings, validation_size):
    reader = Reader(rating_scale=(ratings.rating.min(), ratings.rating.max()))
    data = Dataset.load_from_df(ratings[['sid','pid','rating']], reader)
    
    if validation_size > 0:
        trainset, testset = surprise_split(data, test_size=validation_size, random_state=seed)
        model = SVD(n_factors=n_factors, lr_all=lr, reg_all=reg)
        model.fit(trainset)
        predictions = model.test(testset)
        accuracy.rmse(predictions)
    else:
        trainset = data.build_full_trainset()
        model = SVD(n_factors=n_factors, lr_all=lr, reg_all=reg)
        model.fit(trainset)
    
    return model

def predict(model):
    sub_df = read_sample_df()
    pairs = list(zip(sub_df.sid_pid, sub_df.sid, sub_df.pid))
    
    sid_pid_list, sids, pids = zip(*pairs)
    preds = [model.predict(s, p).est for s, p in tqdm(zip(sids, pids), total=len(sids))]
    clipped = [min(max(r, 1), 5) for r in preds]
    return sid_pid_list, clipped

def run(args):
    mode, arg = args
    
    ratings_df, wishlist_df = read_data_df()
    ext = integrate_wishlist(ratings_df, wishlist_df)
    
    if mode == 'cv':
        print("CV Scores:", cv(ext, arg))
    else:
        model = train(ext, arg)
        if mode == 'predict':
            sid_pid_list, clipped = predict(model)
            out = pd.DataFrame({
                "sid_pid": sid_pid_list,
                "rating": clipped
            })
            fname = f"submission_{ds_name}_svdpp.csv"
            out.to_csv('submissions/'+fname, index=False)
    