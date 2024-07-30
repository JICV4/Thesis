import h5py
import numpy as np
import torch
from torchmetrics import F1Score
from sklearn.preprocessing import LabelEncoder
from torchmetrics import Accuracy
import torch
from maldi_zsl_edit.data import MALDITOFDataModule

def ZSL_levels_metrics(data_path,model,levels,mode="Val"):
    """
    Get a multilevel evaluation and metric of the predictions results of the model
    Inputs:
    - data_path: String with the path to the h5spectra data
    - data_set: Datamodule with the set to evaluate
    - model: ZSL model to get the predictions
    - levels: A list with the granularity levels of the study
    Return:
    - accus: A list with the accuracies of the prediction
    - f1s: A lsit with the f1s of the preductions

    """

    print("--- Getting predictions ---")#Get the predictions
    dm = MALDITOFDataModule( #Personalized lightning data modules
    data_path, #The old has problems on split
    zsl_mode = True, # False: multi-class CLF, True: ZSL
    split_index = 0, # independent train-val-test split numbered 0-9
    batch_size = 128, # important hyperparameter
    n_workers = 2, # you can leave this always if you are not CPU limited
    in_memory = True, # you can leave this always if memory is no problem
    )
    dm.setup(None)
    
    if mode == "Val" : 
        data_set = dm.val_dataloader()
        splits = ['val']
        print("Working with validation set\n")
    elif mode == "Train": 
        data_set = dm.train_dataloader()
        splits = ['train']
        print("Working with train set\n")
    elif mode == "Split":
        data_set = dm.val_dataloader()
        splits = ['val_geni','val_spec','val_strain']
        print("Working with split validation data set\n")

    ev_species = {}
    for split in splits:
        ev_species[split] = [set(),[],[]] #Strain, predict, real

    #for split in ev_species:
    #    minibatch = next(iter(data_set))
    #    y_pred = torch.empty((0,minibatch['seq_ohe'].shape[0])) #the second is the number of species #Change to 788 or 463 for val vs train
    #    y_real= []

    #Get the predictions and order them
    with torch.no_grad():
        for minibatch in iter(data_set): #On the split said if train, val, etc, 
            y_hat = torch.argmax(model(minibatch),axis=1)
            y_real = minibatch['strain']
            #y_hat = model(minibatch)
            #y_pred = torch.cat((y_pred,y_hat),dim=0)
            #y_real+= list(minibatch['strain'])
            for i in range(len(y_real)): #128 is batch size
                split = minibatch["group"][i] 
                if mode == "Val": split = "val"
                ev_species[split][0].add(minibatch['seq_names'][y_real[i]])
                ev_species[split][1].append(y_hat[i]) #= torch.cat((ev_species[split][1] ,y_hat),dim=0)
                ev_species[split][2].append(y_real[i])

    #print(y_pred.shape) #(batch size, total possible species)
    #y_pred
    #Get the multilevel labels and use it for the accu and f1
    
    accus = []
    f1s = [] 
    granularity_lvl = len(levels) 
    for split in ev_species:
        #Get the indexes and stavlis the resolution
        pred_ind = ev_species[split][1]
        real_ind = ev_species[split][2]

        print(f"\n--- Multi level evaluation {split} ---")
        filos = minibatch["seq_names"]
        #Get the multilevel predictions, consider how the data is encoded (genus, species, strain)
        ml_real = []
        ml_pred = []
        for i in range(len(real_ind)): #Iterate all the answer
            #for real:
            s_real = filos[real_ind[i]].split(";") #Use it to get the strain name and split it
            ml_real.append(s_real) #Store the split
            #for pred:
            s_pred = filos[pred_ind[i]].split(";") 
            ml_pred.append(s_pred)

        #Get them on the right format
        ml_real = np.array(ml_real).T
        ml_pred = np.array(ml_pred).T
        #List for better iteratation
        ml_reals = ml_real.tolist()
        ml_preds = ml_pred.tolist()

        #Get all the possible multilevel labels 
        ml_level = []
        for i in range(len(filos)):
            s_level = filos[i].split(";")
            ml_level.append(s_level)
        ml_level = np.array(ml_level).T
        ml_levels = ml_level.tolist()

        #Total number of labels
        for i in range(granularity_lvl):
            n = len(list(set(ml_levels[i])))
            print(f"For {levels[i]} there are {n} different labels")

        print("\n--- Calculating Accuracy ---") # run accu for each level of complexity
        accu_levels = []
        for level in range(granularity_lvl ):
            accu_levels.append(accu_score(ml_reals[level], ml_preds[level], ml_levels[level]))

        # see the results
        for i in range(granularity_lvl ):
            print(f"For the level {levels[i]} the accu score is: {accu_levels[i]}") 
        accus.append(accu_levels)

        print("\n--- Calculating F1 scores ---")# run f1_macro_score for each level of complexity
        F1_levels = []
        for level in range(granularity_lvl ):
            F1_levels.append(f1_macro_score(ml_reals[level], ml_preds[level], ml_levels[level]))

        # see the results
        for i in range(granularity_lvl ):
            print(f"For the level {levels[i]} the F1 score is: {F1_levels[i]}") #The predictions are no the same as the output, maybe F1 is not used there
        f1s.append(F1_levels)

    return accus, f1s
    

def accu_score(y_true, y_pred, level_lab):
    label_encoder = LabelEncoder()
    y_true_encoded = label_encoder.fit_transform(level_lab)
    y_true_encoded = label_encoder.transform(y_true)
    y_pred_encoded = label_encoder.transform(y_pred)

    #Using sklearn
    #accu = accuracy_score(y_true_encoded, y_pred_encoded, normalize=True) #The normalize True = number of correct predictions, False = fraction of correct predictions
    
    #Using torch
    accu = Accuracy(task="multiclass", num_classes=len(set(level_lab))) 
    accu = accu(torch.tensor(y_pred_encoded), torch.tensor(y_true_encoded))
    
    return accu

#Create an F1 evaluator
def f1_macro_score(y_true, y_pred, level_lab): #micro average is basically accuracy
    label_encoder = LabelEncoder()
    y_true_encoded = label_encoder.fit_transform(level_lab)
    y_true_encoded = label_encoder.transform(y_true)
    y_pred_encoded = label_encoder.transform(y_pred)

    #Using sklearn
    #f1_scores = f1_score(y_true_encoded, y_pred_encoded, average=None)
    #macro_f1 = sum(f1_scores) / len(f1_scores)

    #Using torch
    macro_f1 = F1Score(task="multiclass", num_classes=len(set(level_lab)), average='macro') 
    macro_f1 = macro_f1(torch.tensor(y_pred_encoded), torch.tensor(y_true_encoded))

    return macro_f1


