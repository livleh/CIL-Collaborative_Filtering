from config import general_params as params
from models.als import run as als
from models.svd import run as svd
from models.svdpp import run as svdpp
from models.ncf import run as ncf
from models.vae import run as vae
from models.embeddings import run as emb
from models.attention import run as att
from models.deepmatrix import run as mdm
import time
import argparse


models = {
    'als' : als,
    'svd' : svd,
    'svdpp' : svdpp,
    'ncf' : ncf,
    'vae' : vae,
    'embeddings' : emb,
    'attention' : att,
    'deepmatrix': mdm
}

def main():
   
    
    parser = argparse.ArgumentParser(
        description="Run model in cross-validation, training or prediction mode"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="deepmatrix",
        help="Model name or path (default: deepmatrix)"
    )

    parser.add_argument(
        "--mode",
        choices=["cv", "train", "predict"],
        default="predict",
        help="Execution mode: 'cv' or 'train' or 'predict' (default: predict)"
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation (only used if --mode=cv, default: 5)"
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Validation split size (only used if --mode=predict, default: 0.1)"
    )

    args = parser.parse_args()

    print(f"Model: {args.model}")
    model = args.model
    
    if model in models:
        function = models[model]
    else:
        raise Exception('Please choose one of the following models in the config.py file: ' + str.join(", ", models.keys()) )
    
    if model in ['svd', 'svdpp', 'ncf', 'embeddings', 'deepmatrix', 'als', 'vae'] and args.mode == "cv":
        print(f"Running cross-validation with {args.folds} folds")
        pars = args.mode, args.folds
    elif args.mode == "predict":
        print(f"Running prediction with validation split = {args.val_size}")
        pars = args.mode, args.val_size
    elif args.mode == 'train':
        print(f"Running training with validation split = {args.val_size}")
        pars = args.mode, args.val_size
    else:
        raise Exception('Not supported mode for this algorithm')
        
    start_time = time.time()
    
    function(pars)

    print("--- Runtime: %s seconds ---" % (time.time() - start_time))

if __name__=='__main__':
    main()