from lightning import LightningDataModule
import torch
import h5torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd

class MALDITOF_ZSL_Dataset(h5torch.Dataset):
    def __init__(self, file, split_index = 0, in_memory = True, return_all_seqs = True, frac = "train", **kwargs): #the frac arg is redefined when the data set is constructed bellow
        super().__init__(
            file,
            in_memory=in_memory,
            sample_processor=self.sample_processor,
            subset=("0/split_%s" % split_index, frac)
        )
        self.seq = self.f["1/strain_seq_aligned"].view(np.ndarray)
        self.name = self.f["1/strain_names"].view(np.ndarray)
        self.ohe = self.f["1/strain_seq_aligned_ohe"].view(np.ndarray) #added

        self.return_all = return_all_seqs
        self.split = "0/split_%s" % split_index

        self.frac = frac #Added fot ZSL seq
        if self.frac == "train": #Added for ZSL seq, consider one for the val
        #   print("\n create index train\n") 
            self.indices_in_train = np.unique(self.f["central"][:][self.f[self.split][:].astype(str) == "train"].argmax(1))
            self.train_index_mapper = {v : k for k, v in enumerate(self.indices_in_train)}
        if self.frac == "val": #Added (myself) for ZSL seq
        #   print("\n create index val\n") 
            self.indices_in_val = np.unique(self.f["central"][:][np.char.startswith(self.f[self.split][:].astype(str), 'val')].argmax(1))
            #self.indices_in_val = np.unique(self.f["central"][:][self.f[self.split][:].astype(str) == "val"].argmax(1))
            self.val_index_mapper = {v : k for k, v in enumerate(self.indices_in_val)}

    def sample_processor(self, f, sample):
        return {
            "intensity": torch.tensor(sample["0/intensity"]).float(),
            "mz": torch.tensor(sample["0/mz"] if "0/mz" in sample else f["unstructured/mz"][:]),
            "strain" : sample["central"].argmax(),
            "group" : sample[self.split]
        }

    def batch_collater(self, batch):
        batch_collated = {}
        batch_collated["intensity"] = torch.stack([b["intensity"] for b in batch])
        batch_collated["mz"] = torch.stack([b["mz"] for b in batch])
        batch_collated["group"] = [b["group"] for b in batch]
        if self.return_all:
            if self.frac == "train": #Added fot ZSL seq
                #print("\n create values train\n") 
                batch_collated["strain"] = torch.tensor(np.array([self.train_index_mapper[b["strain"]] for b in batch]))
                batch_collated["seq"] = torch.tensor(self.seq[self.indices_in_train]).to(torch.int) #Maybe here is where we need to do the data type
                batch_collated["seq_names"] = list(self.name.astype(str)[self.indices_in_train])
                batch_collated["seq_ohe"] = torch.tensor(self.ohe[self.indices_in_train]).to(torch.float) #added

            elif self.frac == "val": #Added (myself) fot ZSL seq
            #    print("\n create values val\n") 
                batch_collated["strain"] = torch.tensor(np.array([self.val_index_mapper[b["strain"]] for b in batch]))
                batch_collated["seq"] = torch.tensor(self.seq[self.indices_in_val]).to(torch.int)
                batch_collated["seq_names"] = list(self.name.astype(str)[self.indices_in_val])
                batch_collated["seq_ohe"] = torch.tensor(self.ohe[self.indices_in_val]).to(torch.float) #added

            else:
                batch_collated["strain"] = torch.tensor(np.array([b["strain"] for b in batch]))
                batch_collated["seq"] = torch.tensor(self.seq).to(torch.int) #here all the seq are passed, if we dont specify a condition for val then all the seq will be on val
                batch_collated["seq_names"] = list(self.name.astype(str))
                batch_collated["seq_ohe"] = torch.tensor(self.ohe).to(torch.float) #added
        else:
            ixes = np.array([b["strain"] for b in batch])
            batch_collated["strain"] = torch.arange(len(batch_collated["intensity"]))
            batch_collated["seq"] = torch.tensor(self.seq[ixes]).to(torch.int) #Add the .to(torch.int) to try to correct an error
            batch_collated["seq_names"] = list(self.name.astype(str)[ixes])
            batch_collated["seq_ohe"] = torch.tensor(self.ohe[ixes]).to(torch.float) #added
        return batch_collated
    
class MALDITOF_MC_Dataset(h5torch.Dataset):
    def __init__(self, file, split_index = 0, in_memory = True, frac = "train", strain_to_spec_mapping = dict(), **kwargs,):
        super().__init__(
            file,
            in_memory=in_memory,
            sample_processor=self.sample_processor,
            subset=("0/split_%s" % split_index, frac)
        )

        self.strain_to_spec = strain_to_spec_mapping
        self.split = "0/split_%s" % split_index

    def sample_processor(self, f, sample):
        return {
            "intensity": torch.tensor(sample["0/intensity"]).float(),
            "mz": torch.tensor(sample["0/mz"] if "0/mz" in sample else f["unstructured/mz"][:]),
            "species" : self.strain_to_spec[sample["central"].argmax()],
            "group" : sample[self.split]
        }

    def batch_collater(self, batch):
        batch_collated = {}
        batch_collated["intensity"] = torch.stack([b["intensity"] for b in batch])
        batch_collated["mz"] = torch.stack([b["mz"] for b in batch])
        batch_collated["group"] = [b["group"] for b in batch]
        batch_collated["species"] = torch.tensor([b["species"] for b in batch])

        return batch_collated

class MALDITOFDataModule(LightningDataModule):
    def __init__(
        self,
        path,
        batch_size=512,
        n_workers=4,
        in_memory=True,
        split_index = 0,
        return_all_seqs = True,
        zsl_mode = True,
    ):
        super().__init__()
        self.path = path
        self.in_memory = in_memory
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.split = split_index
        self.return_all_seqs = return_all_seqs
        self.zsl = zsl_mode
        self.dataset_type = (MALDITOF_ZSL_Dataset if zsl_mode else MALDITOF_MC_Dataset) #Here we decide if the class (and arguments) are for the ZSL or data type or not

    def setup(self, stage):

        if not self.zsl:
            f = h5torch.File(self.path)
            strains_in_train = np.unique(f["central"][:][f["0/split_%s" % self.split][:].astype(str) == "train"].argmax(1))
            strains_in_val = np.unique(f["central"][:][f["0/split_%s" % self.split][:].astype(str) == "val_strain"].argmax(1))
            strains_in_test = np.unique(f["central"][:][f["0/split_%s" % self.split][:].astype(str) == "test_strain"].argmax(1))
            strains_in_train = np.concatenate([strains_in_train, strains_in_val, strains_in_test])
            species_in_train_all = pd.DataFrame(
                f["1/strain_names"][:][strains_in_train]
                )[0].astype(str).str.split(";", expand=True)[[0, 1, 2]].apply(";".join, axis=1).values
            species_in_train = np.unique(species_in_train_all)
            species_to_ix = {v : k for k, v in enumerate(species_in_train)}
            strain_to_spec = {k : species_to_ix[v] for k, v in zip(strains_in_train, species_in_train_all)}

            self.n_species = len(strain_to_spec)

        self.train = self.dataset_type(
            self.path,
            split_index = self.split,
            in_memory = self.in_memory,
            return_all_seqs = self.return_all_seqs,
            frac = "train",
            strain_to_spec_mapping = (None if self.zsl else strain_to_spec),
        )

        self.val = self.dataset_type(
            self.path,
            split_index = self.split,
            in_memory = self.in_memory,
            return_all_seqs = self.return_all_seqs,
            frac = "val" + ("" if self.zsl else "_strain"),
            strain_to_spec_mapping = (None if self.zsl else strain_to_spec),
        )

        self.test = self.dataset_type(
            self.path,
            split_index = self.split,
            in_memory = self.in_memory,
            return_all_seqs = self.return_all_seqs,
            frac = "test" + ("" if self.zsl else "_strain"),
            strain_to_spec_mapping = (None if self.zsl else strain_to_spec),
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            num_workers=self.n_workers,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.train.batch_collater,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            num_workers=self.n_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.val.batch_collater,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            num_workers=self.n_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.test.batch_collater,
        )