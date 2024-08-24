# Thesis repository

## Usage instructions / guidelines

- A README.md containing thoughts, notes, todos, and where you keep track of things you did in order to make everything reproducible (instructions for how to obtain the exact same data set you obtained, up until the very details of which version of remap, which version of esm, etc .. was used and how they were combined etc)
- LICENSE files and .gitignore files. A standard gitignore should be enough (the one I included is the github default for python), data folders should not be in the gitignore as they should live outside of your github repo anyway. Same goes for notebooks. This is the general structure I follow:
```
/thesis
├── Thesis (The github repository)
│   ├── maldi_zsl
│   │   └── models.py
│   │   └── data.py
│   └── README.md, etc ..
├── notebooks
│   ├── random_stuff.ipynb
│   ├── figure_making.ipynb
│   ├── tuning_results.ipynb
│   ├── idea_number5842.ipynb
├── data
│   ├── datafile.h5t
│   ├── anything_else
├── logs
│   ├── modelcheckpoint1
│   ├── modelcheckpoint2
```
Note that notebooks, data, and model checkpoints are outside of the GitHub repository!
This structure should make sure everything is organized and no files are contaminated to where they should not belong. As the project is structured as a python package and is install in editable/development mode, the exact location of the github repository folder relative to all other files is of no importance.


## Dev install steps:

```
conda create --name env
conda activate env
conda install pip
pip install -e .
```


## Dataset creation

After installing the `maldi_zsl` package:
Create an account on the lpsn website first, then using your credentials, run the script:

```
python Thesis/maldi_zsl/scripts/process_LMUGent.py /home/jorge/thesis/Data/SILVA_138.1_SSURef_NR99_tax_silva_trunc.fasta /home/data/shared/bacteria/labelsshortdb /home/data/shared/bacteria/spectradb your_lpsn_email your_lpsn_password ./zsl_raw.h5t ./zsl_binned.h5t ./align.fa
```

This will create two files, one is the most important: `zsl_binned.h5t`.

Add data split:
```
python Thesis/maldi_zsl/scripts/add_splits.py ./zsl_binned.h5t
```


After running these two scripts you are ready to use the dataloaders:
```python
from maldi_zsl.data import MALDITOFDataModule


dm = MALDITOFDataModule(
    "./zsl_binned.h5t",
    zsl_mode = False, # False: multi-class CLF, True: ZSL
    split_index = 0, # independent train-val-test split numbered 0-9
    batch_size = 16, # important hyperparameter
    n_workers = 2, # you can leave this always if you are not CPU limited
    in_memory = True, # you can leave this always if memory is no problem
    )

dm.setup(None)

batch = next(iter(dm.train_dataloader()))

batch
```

## TO-DO

- [x] Implement Zero shot learning model
- [x] Implement training script for ZSL model
- [x] Implement multi-level accuracy metrics (family-level, genus-level, species-level, strain-level)
