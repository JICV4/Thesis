from lightning import LightningDataModule
import torch
import h5torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def sample_processor(f, sample):
    spectrum = {
        "intensity": torch.tensor(sample["0/intensity"]).float(),
        "mz": torch.tensor(sample["0/mz"] if "0/mz" in sample else f["unstructured/mz"][:]),
        "species" : sample["central"].argmax()
    }

    sample = {k: v for k, v in sample.items() if k not in ["0/intensity", "0/mz", "central"]}
    return sample | spectrum
    
class MaldiZSLCollater():
    def __init__(self, dataset, return_all_seqs = True):
        self.seq = dataset.f["1/strain_seq_aligned"].view(np.ndarray)
        self.name = dataset.f["1/strain_names"].view(np.ndarray)

        self.return_all = return_all_seqs

    def __call__(self, batch):
        batch_collated = {}
        batch_collated["intensity"] = torch.tensor(np.array([b["intensity"] for b in batch]))
        batch_collated["mz"] = torch.tensor(np.array([b["mz"] for b in batch]))

        if self.return_all:
            batch_collated["species"] = torch.tensor(np.array([b["species"] for b in batch]))
            batch_collated["seq"] = torch.tensor(self.seq)
            batch_collated["seq_names"] = list(self.name.astype(str))
        
        else:
            ixes = np.array([b["species"] for b in batch])
            batch_collated["species"] = torch.arange(len(batch_collated["intensity"]))
            batch_collated["seq"] = torch.tensor(self.seq[ixes])
            batch_collated["seq_names"] = list(self.name.astype(str)[ixes])
        return batch_collated


class SpeciesClfDataModule(LightningDataModule):
    def __init__(
        self,
        path,
        batch_size=512,
        n_workers=4,
        in_memory=True,
        zsl_mode=True, # TODO
    ):
        super().__init__()
        self.path = path
        self.in_memory = in_memory
        self.n_workers = n_workers
        self.batch_size = batch_size

    def setup(self, stage):
        f = h5torch.File(self.path)

        if self.in_memory:
            f = f.to_dict()

        self.train = MALDITOFDataset(
            f, subset=("unstructured/split", "train")
        )

        # TODO
        #self.val = MALDITOFDataset( # TODO
        #    f, subset=("unstructured/split", "val")
        #)

        #self.test = MALDITOFDataset(
        #    f, subset=("unstructured/split", "test")
        #)

        self.n_species = len(f["unstructured/species_labels"])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            num_workers=self.n_workers,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=batch_collater,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            num_workers=self.n_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=batch_collater,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            num_workers=self.n_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=batch_collater,
        )


def batch_collater(batch):
    batch_collated = {}
    keys = list(batch[0])
    for k in keys:
        v = [b[k] for b in batch]
        if isinstance(v[0], str):
            batch_collated[k] = v
        elif isinstance(v[0], (int, np.int64)):
            batch_collated[k] = torch.tensor(v)
        elif isinstance(v[0], np.ndarray):
            if len({t.shape for t in v}) == 1:
                batch_collated[k] = torch.tensor(np.array(v))
            else:
                batch_collated[k] = pad_sequence(
                    [torch.tensor(t) for t in v], batch_first=True, padding_value=-1
                )
        elif torch.is_tensor(v[0]):
            if len({t.shape for t in v}) == 1:
                batch_collated[k] = torch.stack(v)
            else:
                if v[0].dtype == torch.bool:
                    batch_collated[k] = pad_sequence(
                        v, batch_first=True, padding_value=False
                    )
                else:
                    batch_collated[k] = pad_sequence(
                        v, batch_first=True, padding_value=-1
                    )
    return batch_collated