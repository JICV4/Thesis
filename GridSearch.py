from lightning.pytorch import Trainer, seed_everything #https://lightning.ai/docs/pytorch/stable/common/trainer.html
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping #3
from lightning.pytorch.loggers import TensorBoardLogger #3
from maldi_zsl.data import MALDITOFDataModule #1
from maldi_zsl.models import ZSLClassifier #2
import os
import sys
import random
from torch import manual_seed as torch_manual_seed
from numpy.random import seed as np_random_seed
from torch.cuda import is_available as torch_cuda_is_available
from torch.cuda import manual_seed as torch_cuda_manual_seed
from torch.cuda import manual_seed_all as torch_cuda_manual_seed_all
from datetime import datetime
from maldi_zsl.utils import ZSL_levels_metrics

def set_seed(seed):
    seed_everything(seed, workers=True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np_random_seed(seed)
    torch_manual_seed(seed)
    if torch_cuda_is_available():
        torch_cuda_manual_seed(seed)
        torch_cuda_manual_seed_all(seed)


def grid_search (config,data_path):
    set_seed(4)
    corpus = """
\\begin{table}
\caption{Results of the hyperparameter tunning considering the general accuracy.}
\centering
\small
\\begin{tabular}{l c c c c c c}
\\toprule
\multicolumn{6}{c}{Parameter} & \multicolumn{1}{c}{Result} \\\\ \cmidrule(lr){1-6} \cmidrule(lr){7-7}
pool & emb\_dim & lr & cnn\_base & kernel & cnn\_hid & Accuracy\\\\
\midrule"""
    timenow = datetime.now()
    strtime = timenow.strftime("%Y-%m-%d_%H-%M-%S")
    dm = MALDITOFDataModule( 
        data_path, 
        zsl_mode = True, # False: multi-class CLF, True: ZSL
        split_index = 1, # 0 for not general eva 1 for general eva, 2 for not general eva of split 1
        batch_size = 512, # important hyperparameter
        n_workers = 2, 
        in_memory = True, 
        general = True # False: Regular ZSL (only val species), True:General ZSL (val+train species)
        )
    dm.setup(None)  
    levels = ["Family", "Genus", "Species", "Strain"]
    best_acc = 0
    best_conf = "None"
    best_model = "None"
    i=0
    print("Starting the hyperparameter tunning")
    print(f"Working with {data_path}")
    print(f"The search space is:\n{config}")
    with open(os.path.join(sys.path[0],f'results/summary_hptune_{strtime}.txt'), 'a') as file:
        file.write(corpus)
    with open(os.path.join(sys.path[0],f'results/hptune_{strtime}.txt'), 'w') as f:
        sys.stdout = f
        print(f"Working with {data_path}")
        print(f"The search space is:\n{config}")
        for pool in config["pool"]:
            for emb_dim in config["emb_dim"]:
                for lr in config["lr"]:
                    #for dropout in config["dropout"]:
                    for cnn_base in config["cnn_base"]:
                        for kernel in config["kernel"]:
                            for cnn_hid in config["cnn_hid"]:
                                vmod = f"model_{i}"
                                current = f"Configuration: {pool}, {emb_dim}, {lr}, {cnn_base}, {kernel}, {cnn_hid}"
                                i+=1
                                if i <= 44: continue #To correct in case the breakloop
                                print(f"\n\n\n---{vmod}---\n")
                                print(current)
                                #Define model                               
                                model = ZSLClassifier(
                                    embed_dim = emb_dim,
                                    cnn_kwargs= { 
                                        'conv_sizes' : cnn_base, 
                                        'hidden_sizes' : cnn_hid,
                                        'blocks_per_stage' : 2,
                                        'kernel_size' : kernel,
                                        'dropout' : 0.2,
                                        'mode': pool, 
                                    },
                                    n_classes = 623,
                                    t_classes = 463,
                                    lr=lr,
                                    weight_decay=0, 
                                    lr_decay_factor=1.00, 
                                    warmup_steps=250,
                                )                            
                            
                                val_ckpt = ModelCheckpoint(monitor="val_acc", mode="max")
                                callbacks = [val_ckpt, EarlyStopping(monitor="val_acc", patience=20, mode="max")]
                                logger = TensorBoardLogger(os.path.join(sys.path[0],"logs"), name=f"hptune_{strtime}", version=vmod)

                                trainer = Trainer(
                                    min_epochs= 80,
                                    max_epochs = 100, 
                                    accelerator='gpu', 
                                    strategy='auto',
                                    callbacks=callbacks,
                                    logger=logger,
                                    devices=[0]) 

                                sys.stdout = sys.__stdout__ #We dont want to record all the training
                                print(vmod)
                                print(current)
                                try: 
                                    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
                                except:
                                    print("Problem in the training, skipping")
                                    with open(os.path.join(sys.path[0],f'results/summary_hptune_{strtime}.txt'), 'a') as file:
                                        row=f"""
{pool} & {emb_dim} & {lr} & {cnn_base} & {kernel} & {cnn_hid} & problem in training \\\\"""
                                        file.write(row)   
                                    sys.stdout = f #now we can keep the records
                                    print("Problem in the training, skipping")
                                    continue
                                sys.stdout = f #now we can keep the records
                                accug, f1g, gen, unacc, snacc, hmean, ev_species, labels = ZSL_levels_metrics(data_path,model,levels,"Val",split_index=1,general=True)
                                row=f"""
{pool} & {emb_dim} & {lr} & {cnn_base} & {kernel} & {cnn_hid} & {round(gen, 4)} \\\\"""
                                with open(os.path.join(sys.path[0],f'results/summary_hptune_{strtime}.txt'), 'a') as file:
                                    file.write(row)
                                if best_acc < gen:
                                    best_acc = gen
                                    best_model = vmod
                                    best_conf = current
                                print(f"The best model is {best_model}, acc:{best_acc}\n{best_conf}")
    
        tail = """
\\bottomrule
\end{tabular}
\caption{Hyperpameter tunning of the data}
\end{table}


"""
        with open(os.path.join(sys.path[0],f'results/summary_hptune_{strtime}.txt'), 'a') as file:
            file.write(tail)
        print("\n\nHyperparameter tuning completed")
        end = datetime.now() - timenow
        total_seconds = end.total_seconds()
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        print(f"{int(hours)} hours and {int(minutes)} minutes")
        
        sys.stdout = sys.__stdout__
        with open(os.path.join(sys.path[0],f'results/summary_hptune_{strtime}.txt'), 'a') as file:
            file.write(f"Working with {data_path}\nCompleted at {int(hours)} hours and {int(minutes)} minutes")

    print("Finished")

if __name__ == "__main__":
#    config = {
#        "emb_dim": [1024],
#        "lr": [1e-4],
#        "dropout": [0.2],
#        "cnn_base" : [[64,128]],
#        "kernel" : [7],
#        "cnn_hid" : [[128]],
#        "pool": ["max","mean"],
#    }

    config = {
        "emb_dim": [524,1024],
        "lr": [1e-4, 1e-3],
        "cnn_base" : [[32,64],[64,128]],
        "kernel" : [5,7],
        "cnn_hid" : [[0],[64]],
        "pool": ["max","mean"],
    }

    data_path = "Data/zsl_SINAwasabi.h5t"# "../Data/final/zsl_mafft.h5t"
    grid_search (config,os.path.join(sys.path[0],data_path))