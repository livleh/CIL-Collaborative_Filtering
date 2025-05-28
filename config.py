general_params = {}
svd_params = {}
svdpp_params = {}
als_params = {}
ncf_params = {}
vae_params = {}
embeddings_params = {}
attention_params = {}
mdm_params = {}

general_params['data_dir'] = "data/"
general_params['submissions_dir'] = "submissions/"
general_params['train_data_path'] = "data/train_ratings.csv"
general_params['train_tbr_path'] = "data/train_tbr.csv"
general_params['sample_submission_path']= "data/sample_submission.csv"
general_params['n_users'] = 10000
general_params['n_papers'] = 1000
general_params['seed'] = 42

als_params['factor'] = 10
als_params['lambda_reg'] = 1
als_params['num_iterations'] = 10

svd_params['ds_name'] = 'row'
svd_params['n_factors'] = 100
svd_params['lr'] = 0.005
svd_params['reg'] = 0.02

svdpp_params['ds_name'] = 'row'
svdpp_params['n_factors'] = 50
svdpp_params['lr'] = 0.005
svdpp_params['reg'] = 0.02

vae_params['ds_name'] = 'row'
vae_params['batch_size']= 64
vae_params['lr']= 0.001

ncf_params['ds_name'] = 'row'
ncf_params['loss_type']= "cross_entropy"
ncf_params['embed_size']= 16
ncf_params['n_epochs']= 20
ncf_params['lr']= 0.01
ncf_params['lr_decay']= False
ncf_params['epsilon']= 0.00001
ncf_params['reg']= None
ncf_params['batch_size']= 256
ncf_params['sampler']=  "random"
ncf_params['num_neg']= 1
ncf_params['use_bn']= True

embeddings_params['n_epochs'] = 5
embeddings_params['dim'] = 32
embeddings_params['lr'] = 1e-3
embeddings_params['batch_size'] = 64

attention_params['sid_wishlist_size'] = 34
attention_params['sid_context_size'] = 30
attention_params['pid_wishlist_size'] = 50
attention_params['pid_context_size'] = 100
attention_params['embedding_dim'] = 16
attention_params['num_heads'] = 4
attention_params['dropout_rate'] = 0.3
attention_params['l2_reg'] = 1e-4
attention_params['lr'] = 1e-3
attention_params['batch_size'] = 64
attention_params['n_epochs'] = 10

mdm_params['n_epochs'] = 100
mdm_params['batch_size'] = 1024
mdm_params['dim'] = 35
mdm_params['hidden_dim'] = 35
mdm_params['lr'] = 5*1e-5