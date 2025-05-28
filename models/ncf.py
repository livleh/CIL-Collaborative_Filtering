import tensorflow as tf
import pandas as pd
from libreco.algorithms import NCF as libNCF
from libreco.data import DatasetPure, random_split
from libreco.evaluation import evaluate as lib_eval
from data_handler.data_handler import *
import os
from sklearn.model_selection import KFold
from tqdm.auto import tqdm


from config import ncf_params as params, general_params
SEED = general_params['seed']
ds_name = params['ds_name'] 
loss_type = params['loss_type']
embed_size = params['embed_size']
n_epochs = params['n_epochs']
lr = params['lr']
lr_decay = params['lr_decay']
epsilon = params['epsilon']
reg = params['reg']
batch_size = params['batch_size']
sampler = params['sampler']
num_neg = params['num_neg']
use_bn = params['use_bn']


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

def cv(ratings, n_splits=5):
    df = ratings.copy()
    df = df.rename(columns={'sid': 'user', 'pid': 'item', 'rating':'label'})

    rmses = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for i, (train_idx, val_idx) in enumerate(kf.split(df), 1):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        tf.compat.v1.reset_default_graph()
        train_data, train_data_info = DatasetPure.build_trainset(train_df)
        test_data = DatasetPure.build_testset(val_df)

        model = libNCF('rating', train_data_info, loss_type=loss_type, embed_size=embed_size, n_epochs=n_epochs, lr=lr, lr_decay=lr_decay, epsilon=epsilon, reg=reg, batch_size=batch_size, sampler=sampler, num_neg=num_neg, use_bn=use_bn)
      
        model.fit(train_data, False, verbose=2, metrics=['rmse'] )
        eval_result = lib_eval(model, test_data, False, metrics=['rmse'])
        rmses.append(eval_result['rmse'])
    return np.mean(rmses), np.std(rmses)

def train(ratings: pd.DataFrame, validation_size):
    df = ratings.copy()
    df = df.rename(columns={'sid': 'user', 'pid': 'item', 'rating':'label'})
    tf.compat.v1.reset_default_graph()
    
    if validation_size > 0:
        train_data2, eval_data2 = random_split(df, test_size=0.2, seed=SEED)
        train_data, train_data_info = DatasetPure.build_trainset(train_data2)
        eval_data = DatasetPure.build_evalset(eval_data2)
        model = libNCF('rating', train_data_info, loss_type=loss_type, embed_size=embed_size, n_epochs=n_epochs, lr=lr, lr_decay=lr_decay, epsilon=epsilon, reg=reg, batch_size=batch_size, sampler=sampler, num_neg=num_neg, use_bn=use_bn)
        model.fit(train_data, False, verbose=2, metrics=['rmse'], eval_data=eval_data)
    else:
        train_data, train_data_info = DatasetPure.build_trainset(df)
        model = libNCF('rating', train_data_info, loss_type=loss_type, embed_size=embed_size, n_epochs=n_epochs, lr=lr, lr_decay=lr_decay, epsilon=epsilon, reg=reg, batch_size=batch_size, sampler=sampler, num_neg=num_neg, use_bn=use_bn)
        model.fit(train_data, False, verbose=2, metrics=['rmse'])
    return model

def predict(model: libNCF):
    sub_df = read_sample_df()
    pairs = list(zip(sub_df.sid_pid, sub_df.sid, sub_df.pid))
    
    sid_pid_pairs, sids, pids = zip(*pairs)
    print("Batch prediction started...")
    try:
        preds = model.predict(sids, pids)
    except:
        preds = [model.predict(s, p) for s, p in tqdm(zip(sids, pids), total=len(sids))]
    clipped = [min(max(r, 1), 5) for r in preds]
    return sid_pid_pairs, clipped

def run(args):
    mode, arg = args
    
    ratings_df, wishlist_df = read_data_df()
    ext = integrate_wishlist(ratings_df, wishlist_df)
        
    if mode == 'cv':
        print(cv(ext, arg))
    else:
        model = train(ext, arg)
        if mode == 'predict':
            sid_pid_list, clipped = predict(model)
            out = pd.DataFrame({
                "sid_pid": sid_pid_list,
                "rating": clipped
            })
            fname = f"submission_{ds_name}_ncf.csv"
            out.to_csv('submissions/'+fname, index=False)
    