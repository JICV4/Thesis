import h5torch
import pandas as pd
import numpy as np

f = h5torch.File("./zsl_binned.h5t")
taxonomy = pd.Series(f["1/strain_names"][:].astype(str)).str.split(";", expand=True)

for split_number in range(10):

    genuses_train = []
    genuses_val = []
    genuses_test = []
    for family in np.unique(taxonomy[0]): # to split genuses, look at every family
        # unique genuses in that family
        genuses_in_family = np.unique(taxonomy[taxonomy[0] == family][[0,1]].apply(";".join, axis=1).index.values) 
        if len(genuses_in_family) == 1: #if only one recorded genus in the family with data: all to train
            genuses_train.append(genuses_in_family[0])
        else: #else, split.
            to_train, to_eval = np.split(np.random.permutation(genuses_in_family), [int(len(genuses_in_family) * 0.8)])

            if np.random.rand() > 0.5:
                to_val, to_test = to_eval[:len(to_eval)//2], to_eval[len(to_eval)//2:]
            else:
                to_test, to_val = to_eval[:len(to_eval)//2], to_eval[len(to_eval)//2:]
            
            for t in to_train: genuses_train.append(t)
            for t in to_val: genuses_val.append(t)
            for t in to_test: genuses_test.append(t)

    # now do the same for species splitting
    taxonomy_sub_train_genuses = taxonomy.loc[genuses_train]
    genuses_names = taxonomy_sub_train_genuses[[0,1]].apply(";".join, axis=1)
    species_train = []
    species_val = []
    species_test = []
    for genus in np.unique(genuses_names): # to split species, look at every genus
        # unique genuses in that family
        species_in_genus = np.unique(taxonomy_sub_train_genuses[genuses_names == genus][[0,1,2]].apply(";".join, axis=1).index.values)
        if len(species_in_genus) == 1: #if only one recorded species in the genus with data: all to train
            species_train.append(species_in_genus[0])
        else: #else, split.
            to_train, to_eval = np.split(np.random.permutation(species_in_genus), [int(len(species_in_genus) * 0.8)])

            if np.random.rand() > 0.5:
                to_val, to_test = to_eval[:len(to_eval)//2], to_eval[len(to_eval)//2:]
            else:
                to_test, to_val = to_eval[:len(to_eval)//2], to_eval[len(to_eval)//2:]
            
            for t in to_train: species_train.append(t)
            for t in to_val: species_val.append(t)
            for t in to_test: species_test.append(t)


    # and the same for strains
    taxonomy_sub_train_species = taxonomy.loc[species_train]
    species_names = taxonomy_sub_train_species[[0,1,2]].apply(";".join, axis=1)
    strains_train = []
    strains_val = []
    strains_test = []
    for species in np.unique(species_names): # to split genuses, look at every family
        # unique genuses in that family
        strains_in_species = np.unique(taxonomy_sub_train_species[species_names == species][[0,1,2,3]].apply(";".join, axis=1).index.values)
        if len(strains_in_species) == 1: #if only one recorded genus in the family with data: all to train
            strains_train.append(strains_in_species[0])
        else: #else, split.
            to_train, to_eval = np.split(np.random.permutation(strains_in_species), [int(len(strains_in_species) * 0.8)])

            if np.random.rand() > 0.5:
                to_val, to_test = to_eval[:len(to_eval)//2], to_eval[len(to_eval)//2:]
            else:
                to_test, to_val = to_eval[:len(to_eval)//2], to_eval[len(to_eval)//2:]
            
            for t in to_train: strains_train.append(t)
            for t in to_val: strains_val.append(t)
            for t in to_test: strains_test.append(t)

    strain_ids = f["central"][:].argmax(1)
    split = np.empty_like(strain_ids, dtype="object")
    split[np.isin(strain_ids, genuses_train)] = "train"
    split[np.isin(strain_ids, genuses_val)] = "val_geni"
    split[np.isin(strain_ids, genuses_test)] = "test_geni"

    split[np.isin(strain_ids, species_train)] = "train"
    split[np.isin(strain_ids, species_val)] = "val_spec"
    split[np.isin(strain_ids, species_test)] = "test_spec"

    split[np.isin(strain_ids, strains_train)] = "train"
    split[np.isin(strain_ids, strains_val)] = "val_strain"
    split[np.isin(strain_ids, strains_test)] = "test_strain"

    np.savetxt("./Thesis/maldi_zsl/utils/split_%s.txt" % split_number, split.astype(str), delimiter=" ", fmt="%s")