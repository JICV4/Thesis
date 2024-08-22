import h5py
import numpy as np
import torch
from torchmetrics import F1Score
from sklearn.preprocessing import LabelEncoder
from torchmetrics import Accuracy
import torch
from maldi_zsl_edit.data import MALDITOFDataModule
from statistics import harmonic_mean
import torch.nn as nn



def ZSL_levels_metrics(data_path,model,levels,mode="val",split_index=1,general=True):
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
    split_index = split_index, # independent train-val-test split numbered 0-9
    batch_size = 1024, # important hyperparameter
    n_workers = 2, # you can leave this always if you are not CPU limited
    in_memory = True, # you can leave this always if memory is no problem
    general = general
    )
    dm.setup(None)
    
    if mode == "Val" : 
        data_set = dm.val_dataloader()
        if split_index == 1:
            splits = ['general','val_geni','val_spec','val_strain','val_unseen','val_seen']
        else:
            splits = ['general','val_geni','val_spec','val_strain']
        print("Working with validation set\n")
    elif mode == "Train": 
        data_set = dm.train_dataloader()
        splits = ['train']
        print("Working with train set\n")
    elif mode == "Test":
        data_set = dm.test_dataloader()
        if split_index == 1:
            splits = ['general','test_geni','test_spec','test_strain','test_unseen','test_seen']
        else:
            splits = ['general','test_geni','test_spec','test_strain']
        print("Working with test set\n")

    labels = next(iter(data_set))["seq_names"]
    n_species = len(labels)
    ev_species = {}
    for split in splits:
        ev_species[split] = [set(),[],[],torch.empty((0,n_species)),[]] #Strain, predict, real, score, prob

    #Get the predictions and order them
    with torch.no_grad():
        for minibatch in iter(data_set): #On the split said if train, val, etc, 
            y_score = model(minibatch)
            y_hat = torch.argmax(y_score,axis=1)
            y_real = minibatch['strain']
            for i in range(len(y_real)): 
                split = minibatch["group"][i] 
                ev_species[split][0].add(minibatch['seq_names'][y_real[i]])
                ev_species[split][1].append(y_hat[i]) #= torch.cat((ev_species[split][1] ,y_hat),dim=0)
                ev_species[split][2].append(y_real[i])
                ev_species[split][3] = torch.cat((ev_species[split][3],y_score[i:i+1,:]),dim=0)
                if mode != "Train":
                    ev_species['general'][0].add(minibatch['seq_names'][y_real[i]])
                    ev_species['general'][1].append(y_hat[i]) #= torch.cat((ev_species[split][1] ,y_hat),dim=0)
                    ev_species['general'][2].append(y_real[i])
                    ev_species['general'][3] = torch.cat((ev_species['general'][3],y_score[i:i+1,:]),dim=0)
                    if split_index == 1 and mode == "Val" and split != "val_seen":
                        ev_species['val_unseen'][0].add(minibatch['seq_names'][y_real[i]])
                        ev_species['val_unseen'][1].append(y_hat[i]) #= torch.cat((ev_species[split][1] ,y_hat),dim=0)
                        ev_species['val_unseen'][2].append(y_real[i])
                        ev_species['val_unseen'][3] = torch.cat((ev_species['val_unseen'][3],y_score[i:i+1,:]),dim=0)
                    if split_index == 1 and mode == "Test" and split != "test_seen":
                        ev_species['test_unseen'][0].add(minibatch['seq_names'][y_real[i]])
                        ev_species['test_unseen'][1].append(y_hat[i]) #= torch.cat((ev_species[split][1] ,y_hat),dim=0)
                        ev_species['test_unseen'][2].append(y_real[i])
                        ev_species['test_unseen'][3] = torch.cat((ev_species['test_unseen'][3],y_score[i:i+1,:]),dim=0)


    softmax = nn.Softmax(dim=1)
    for split in ev_species:
        ev_species[split][4] = softmax(ev_species[split][3]) 


    # To store the values
    accus = []
    f1s = [] 
    granularity_lvl = len(levels) 
    c = 0

    unacc = None
    snacc = None
    hmean = None

    for split in ev_species:
        #Get the indexes and stavlis the resolution
        pred_ind = ev_species[split][1]
        real_ind = ev_species[split][2]

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

        #if c == 0: 
        c=1 #Only print this once
            #Total number of labels
        for i in range(granularity_lvl):
            n = len(list(set(ml_levels[i])))
            print(f"For {levels[i]} there are {n} different labels")

        print(f"\n--- Multi level evaluation {split} ---")
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

    if split_index == 1:
        print("\n--- Summary validation ---")
        gen = float(accus[0][-1])
        unacc = float(accus[-2][-1])
        snacc = float(accus[-1][-1])
        hmean = harmonic_mean([unacc,snacc])
        print(f"Unseen acc: {unacc}, Seen acc: {snacc}, Harmonic mean: {hmean}")


        #return accus, f1s, gen, unacc, snacc, hmean, ev_species, labels
    
    gen = float(accus[0][-1])
    return accus, f1s, gen, unacc, snacc, hmean, ev_species, labels
    

def accu_score(y_true, y_pred, level_lab):
    label_encoder = LabelEncoder()
    y_true_encoded = label_encoder.fit_transform(level_lab)
    y_true_encoded = label_encoder.transform(y_true)
    y_pred_encoded = label_encoder.transform(y_pred)
    accu = Accuracy(task="multiclass", num_classes=len(set(level_lab))) 
    accu = accu(torch.tensor(y_pred_encoded), torch.tensor(y_true_encoded))
    return accu

#Create an F1 evaluator
def f1_macro_score(y_true, y_pred, level_lab): #micro average is basically accuracy
    label_encoder = LabelEncoder()
    y_true_encoded = label_encoder.fit_transform(level_lab)
    y_true_encoded = label_encoder.transform(y_true)
    y_pred_encoded = label_encoder.transform(y_pred)
    macro_f1 = F1Score(task="multiclass", num_classes=len(set(level_lab)), average='macro') 
    macro_f1 = macro_f1(torch.tensor(y_pred_encoded), torch.tensor(y_true_encoded))
    return macro_f1

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch import argmax

def accu_rejt(ev_species, labels, split, pred_res=False, quality=False, decoy_prob=None):
    ts = []
    taccus = []
    fps = []
    rejects = []
    total = len(ev_species[split][4])
    resolution = 100
    decoys = []
    optimum = 0
    balance = 0
    class_qualities = []
    rej_qualities = []
    for t in range(resolution+1):
        t = t/resolution
        ts.append(t)
        rejected = 0
        correct = 0
        pass_dec = 0
        correct_rej = 0
        wrong_rej = 0
        if decoy_prob != None:
            for probs in decoy_prob:
                if max(probs) >= t: pass_dec+=1
            decoys.append(pass_dec/700)
        for i, probs in enumerate(ev_species[split][4]):
            pred = labels[argmax(probs)] #Full label of the prediction
            real = labels[ev_species[split][2][i]] #Full label of the real
            if pred_res != False:
                pred = pred.split(";")[pred_res]
                real = real.split(";")[pred_res]
            if max(probs) >= t:
                #accu_score(real,pred,lab)
                if pred == real: correct+=1
            else:
                rejected+=1
                if pred != real: correct_rej += 1
                else: wrong_rej += 1

        try : accu = correct/(total-rejected)
        except : accu = 0
        taccus.append(accu)
        reject = rejected/total
        rejects.append(reject)
        if accu > reject: optimum = t
        fp = 1-accu
        fps.append(fp)
        if fp > accu: balance = t
        class_qualities.append((correct+correct_rej)/total)
        if wrong_rej == 0 : rej_qualities.append((total-1)*(total-1)+1)
        else : rej_qualities.append((correct_rej/wrong_rej)/(taccus[0]/fps[0]))




    x = np.array(ts)
    y0 = np.array(taccus)
    y1 = np.array(rejects)
    #y2 = np.array(fps)
    y4 = np.array(class_qualities)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 6))

    sns.lineplot(x=x, y=y0, label='Accuracy', color='green')
    sns.lineplot(x=x, y=y1, label='Rejection', color='blue')
    #sns.lineplot(x=x, y=y2, label='FPs', color='red')
    if quality : sns.lineplot(x=x, y=y4, label='Quality', color='purple')

    if decoy_prob != None:
        y2 = np.array(decoys)
        sns.lineplot(x=x, y=y2, label='Decoys', color='red')

    #plt.axvline(x=balance, color='black', linestyle='--', label=f'Balance = {balance}')
    #plt.axvline(x=optimum, color='black', linestyle='--', label=f'Optimum = {optimum}')

    plt.xlabel('Threshold')
    plt.ylabel('Percentage')
    plt.title(f'Accuracy-Rejection Plot: {split}')
    plt.legend()

    plt.show()

    return ts, taccus, class_qualities, rej_qualities

def plot_distribution(data, title="Distribution"):
    sns.set_style(style="whitegrid") 
    sns.histplot(data, bins=50, color='blue', kde=True, alpha=0.7) 
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True)
    plt.show()


def distributions(ev_species,split):
    best = ev_species[split][1]
    y_real = ev_species[split][2]
    true_prob = []
    false_prob = []
    true_score = []
    false_score = []
    for i in range(len(y_real)):
        if best[i] == y_real[i]: 
            true_prob.append(max(ev_species[split][4][i]))
            true_score.append(max(ev_species[split][3][i]))
        else: 
            false_prob.append(max(ev_species[split][4][i]))
            false_score.append(max(ev_species[split][3][i]))  
    
    plot_distribution(np.array(true_prob),"True prob")
    plot_distribution(np.array(false_prob), "False prob")
    plot_distribution(np.array(true_score), "True scores")
    plot_distribution(np.array(false_score), "False scores")