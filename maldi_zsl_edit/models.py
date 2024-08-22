import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lightning import LightningModule
from torchmetrics.classification import MulticlassAccuracy
#from maldi_zsl.blocks import ResidualBlock, GlobalPool, Permute #Maybe just blocks as maldi_zsl is for the installed package
from maldi_zsl_edit.blocks import ResidualBlock, GlobalPool, Permute #Maybe just blocks as maldi_zsl is for the installed package

class MLP(nn.Module):
    def __init__(
        self,
        n_inputs=6000,
        n_outputs=512,
        layer_dims=[512, 512, 512],
        layer_or_batchnorm="layer",
        dropout=0.2,
    ):
        super().__init__()

        c = n_inputs
        layers = []
        for i in layer_dims:
            layers.append(nn.Linear(c, i))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            layers.append(
                nn.LayerNorm(i) if layer_or_batchnorm == "layer" else nn.BatchNorm1d(i)
            )
            c = i

        layers.append(nn.Linear(c, n_outputs))

        self.net = nn.Sequential(*layers)

        self.hsize = n_outputs

    def forward(self, spectrum):
        return self.net(spectrum["intensity"]) #Run the data to the network, as we work with our special format file then we can use the whole h5 file and specify what value is the one we are going to use 

class SpeciesClassifier(LightningModule):
    def __init__(
        self,
        mlp_kwargs={},
        lr=1e-4,
        weight_decay=0,
        lr_decay_factor=1.00,
        warmup_steps=250,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.spectrum_embedder = MLP(**mlp_kwargs)
        n_classes = mlp_kwargs["n_outputs"]

        self.accuracy = MulticlassAccuracy(num_classes=n_classes, average="micro")
        self.top5_accuracy = MulticlassAccuracy(
            num_classes=n_classes, top_k=5, average="micro"
        )

    def forward(self, batch):
        return self.spectrum_embedder(batch)

    def training_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        logits = self(batch) #The model is based on the MLP, that specify the 'intensity' as x so you can use the whole batch as input

        loss = F.cross_entropy(logits, batch["species"]) 

        self.log("train_loss", loss, batch_size=len(batch["species"]))
        return loss

    def validation_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        logits = self(batch) #This is model(batch) so is doing the predictions

        loss = F.cross_entropy(logits, batch["species"])

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["species"]),
        )

        self.accuracy(logits, batch["species"])
        self.log(
            "val_acc",
            self.accuracy,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["species"]),
        )
        self.top5_accuracy(logits, batch["species"])
        self.log(
            "val_top5_acc",
            self.top5_accuracy,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["species"]),
        )

    def predict_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        logits = self(batch)

        return (
            torch.stack([batch["species"].to(logits).unsqueeze(-1), logits], -1),
            batch["0/loc"],
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lambd = lambda epoch: self.hparams.lr_decay_factor
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lambd
        )
        return [optimizer], [lr_scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        # warm up lr
        if self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams.warmup_steps
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def n_params(self):
        params_per_layer = [(name, p.numel()) for name, p in self.named_parameters()]
        total_params = sum(p.numel() for p in self.parameters())
        params_per_layer += [("total", total_params)]
        return params_per_layer



class CNN(nn.Module):
    def __init__(
        self,
        channel = [5], #Each dimension is a letter (-,A,T,C,G)
        n_outputs = 64,
        hidden_sizes = [16, 32, 64, 128], #Out chanels of the convolutions, the first work as embeding dimension
        #Note: The last element of the hidden_sizes should be unique
        blocks_per_stage = 2, #How many residual blocks are applied before the pooling
        kernel_size = 7,
        dropout = 0.2,
    ):
        super().__init__()
        layers = []
        hidden_sizes = channel + hidden_sizes
        for i in range(len(hidden_sizes)): #Next the convolutions begins
            hdim = hidden_sizes[i]
            for _ in range(blocks_per_stage): #The convolutions are added to the model as residual blocks, here two consecutive blocks are added
                layers.append(ResidualBlock(hidden_dim = hdim, kernel_size = kernel_size, dropout = dropout))
            
            if hdim != hidden_sizes[-1]: #After the residual blocks apply a regular convolution to reduce the dimension
                layers.append(nn.Conv1d(hdim, hidden_sizes[i+1], kernel_size = 2, stride = 2))
            else: #If ypu are on the last convolution apply a global pooling instead
                layers.append(GlobalPool(pooled_axis = 2, mode = "max")) #Apply a maxpooling at the end instead of a global pooling?
                #This is for ZSL, here maybe not needed 
                #layers.append(GlobalPool(pooled_axis = 2, mode = "max")) #The globalpool get one value, so the output goes to one neuron and not a 
        
        #Here you can add the MLP dense layers after the convolution, we use later an mlp so dont add it here
        #layers.append(nn.Linear(hidden_sizes[-1], n_outputs)) #End with the activation function

        self.net = nn.Sequential(*layers)
        self.hsize = n_outputs

    def forward(self, batch): #The batch is our perzonalized h5 data set
        # dna should be [B, L], with B=number of samples and L=sequence length, 
        # and should be filled with integers from 0 to 4, encoding the DNA sequence.
        return self.net(batch['seq']) # output will be [B, n_outputs]


class SeqClassifier(LightningModule):
    def __init__(
        self,
        cnn_kwargs={},
        mlp_kwargs={},
        lr=1e-4,
        weight_decay=0,
        lr_decay_factor=1.00,
        warmup_steps=250,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.seq_conv = CNN(**cnn_kwargs)
        self.seq_embedder = MLP(**mlp_kwargs)
        n_classes = mlp_kwargs["n_outputs"]

        self.accuracy = MulticlassAccuracy(num_classes=n_classes, average="micro")
        self.top5_accuracy = MulticlassAccuracy(
            num_classes=n_classes, top_k=5, average="micro"
        )

    def forward(self, batch):
        return self.seq_embedder(self.seq_conv(batch)) #data is already one hot

    def SeqsOneHotEncoder(sequences):
        # Data is already ohe
        """
        One hot encoder for DNA/RNA sequences
        A,a first col
        C,c second col
        G,g third col
        T,t/U,u fourth col
        Others fifth col
        """
        ohe_seqs = []
        mapping = {"A": [1,0,0,0,0], "C": [0,1,0,0,0], "G": [0,0,1,0,0], "T": [0,0,0,1,0], "U": [0,0,0,1,0],
                   "a": [1,0,0,0,0], "c": [0,1,0,0,0], "g": [0,0,1,0,0], "t": [0,0,0,1,0], "u": [0,0,0,1,0]}
        for sequence in sequences:
            ohe_seq = []
            for i in sequence:
                ohe_seq.append(mapping[i] if i in mapping.keys() else [0,0,0,0,1])
            ohe_seqs.append(np.array(ohe_seq).T) #Use the transpose to change the OHE format
        return  np.array(ohe_seqs) #Check format with shape should be (n,c,l), where are instances, channels and length, respectively"

    def training_step(self, batch, batch_idx):
        # Here you can do the one hot encoded
        #batch['seq'] = self.SeqsOneHotEncoder(batch['seq'])
        batch['seq'] = batch['seq'].to(self.dtype)

        logits = self(batch) #The model is based on the MLP, that specify the 'intensity' as x so you can use the whole batch as input

        loss = F.cross_entropy(logits, batch["species"])

        self.log("train_loss", loss, batch_size=len(batch["species"]))
        return loss

    def validation_step(self, batch, batch_idx):
        # Here you can do the one hot encoded
        #batch['seq'] = self.SeqsOneHotEncoder(batch['seq'])
        batch['seq'] = batch['seq'].to(self.dtype)

        logits = self(batch) #This is model(batch) so is doing the predictions

        loss = F.cross_entropy(logits, batch["species"])

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["species"]),
        )

        self.accuracy(logits, batch["species"])
        self.log(
            "val_acc",
            self.accuracy,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["species"]),
        )
        self.top5_accuracy(logits, batch["species"])
        self.log(
            "val_top5_acc",
            self.top5_accuracy,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["species"]),
        )

    def predict_step(self, batch, batch_idx):
        # Here you can do the one hot encoded
        #batch['seq'] = self.SeqsOneHotEncoder(batch['seq']) #Data is already ohe?
        batch['seq'] = batch['seq'].to(self.dtype)

        logits = self(batch)

        return (
            torch.stack([batch["species"].to(logits).unsqueeze(-1), logits], -1),
            batch["0/loc"],
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lambd = lambda epoch: self.hparams.lr_decay_factor
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lambd
        )
        return [optimizer], [lr_scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        # warm up lr
        if self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams.warmup_steps
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def n_params(self):
        params_per_layer = [(name, p.numel()) for name, p in self.named_parameters()]
        total_params = sum(p.numel() for p in self.parameters())
        params_per_layer += [("total", total_params)]
        return params_per_layer

    def get_activations(self, x):
        activations = []

        def hook(module, input, output):
            activations.append(output)

        for name, module in self.named_modules():
            if isinstance(module, torch.nn.modules.Linear): 
                module.register_forward_hook(hook)

        self.eval()
        with torch.no_grad():
            self(x)

        return activations



















class MLPEmbedding(nn.Module):
    def __init__(
        self,
        n_inputs=6000,
        emb_dim = 520,
        layer_dims=[512, 256], #The last layer is the number of embedding dim of the shared space and score function
        layer_or_batchnorm="layer",
        dropout=0.2,
    ):
        super().__init__()
        layer_dims.append(emb_dim)
        c = n_inputs #Inputs of each layer, for the first one is just the number of feathures
        layers = []
        for i in layer_dims:
            layers.append(nn.Linear(c, i))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            layers.append(
                nn.LayerNorm(i) if layer_or_batchnorm == "layer" else nn.BatchNorm1d(i)
            )
            c = i

        layers.append(nn.Linear(c, emb_dim)) #Last layer is omited or used for the embdding

        self.net = nn.Sequential(*layers)

    def forward(self, spectrum):
        return self.net(spectrum["intensity"]) #Run the data to the network, as we work with our special format file then we can use the whole h5 file and specify what value is the one we are going to use 


class CNNEmbedding(nn.Module):
    def __init__(self,
        vocab_size = 6, #Number of words, in this case is 5 as (A,T,C,G,-), apparently the standart is (AGCT), 6 as we separate gaps from unknown
        emb_dim = 520,
        conv_sizes = [32, 64, 128], #Out chanels of the convolutions, note that the embedding vocab is always five and is defined in the layers and not in the convolution
        #Also, the embedding dim shold be here as is the out of the lineal activation function
        blocks_per_stage = 0, #How many residual blocks are applied before the pooling
        kernel_size = 3,
        hidden_sizes = [0], #For an extra mlp at the end
        dropout = 0.2,
        mode = "max"
    ):
        super().__init__()

        layers = [nn.Conv1d(vocab_size, conv_sizes[0], kernel_size = 3, stride = 1)] 

        for i in range(len(conv_sizes)): #Next the convolutions begins
            hdim = conv_sizes[i]
            for _ in range(blocks_per_stage): #The convolutions are added to the model as residual blocks, here two consecutive blocks are added
                layers.append(ResidualBlock(hidden_dim = hdim, kernel_size = kernel_size, dropout = dropout))
            
            if hdim != conv_sizes[-1]: #After the residual blocks apply a regular convolution to reduce the dimension
                layers.append(nn.Conv1d(hdim, conv_sizes[i+1], kernel_size = 2, stride = 1))
            else: #If ypu are on the last convolution apply a global pooling instead
                layers.append(GlobalPool(pooled_axis = 2, mode = mode)) #Check if the dim is the right one, should be the lenght (consider shape)
        
        #Here start the loop of the MLP 
        if hidden_sizes == [0]: #Add the [0] as parameter instead of False
            layers.append(nn.Linear(conv_sizes[-1], emb_dim)) #End with just an activation layer as it is embedding

        else:
            c = conv_sizes[-1]
            #layers.append(nn.Linear(conv_sizes[-1], hidden_sizes[0]))
            for i in hidden_sizes:
                layers.append(nn.Linear(c, i))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                layers.append(nn.LayerNorm(i))
                c = i
            layers.append(nn.Linear(c, emb_dim)) #End with just an activation layer as it is embedding 
        
        self.net = nn.Sequential(*layers) 
        self.hsize = emb_dim
        #self.nlp = nlp #try

    def forward(self, batch): #The batch is our perzonalized h5 data set
        inputs = batch['seq_ohe']
        return self.net(inputs) 

class ZSLClassifier(LightningModule):
    def __init__(
        self,
        embed_dim = 520,
        mlp_kwargs={},
        cnn_kwargs={},
        n_classes = 400,
        t_classes = 400,
        lr=1e-4,
        weight_decay=0,
        lr_decay_factor=1.00,
        warmup_steps=250,
        nlp = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        mlp_kwargs['emb_dim'] = self.embed_dim
        cnn_kwargs['emb_dim'] = self.embed_dim
        #cnn_kwargs['nlp'] = nlp #Try
        self.save_hyperparameters()
        self.spectrum_embedder = MLPEmbedding(**mlp_kwargs)
        self.seq_embedder = CNNEmbedding(**cnn_kwargs)
        self.accuracy = MulticlassAccuracy(num_classes=n_classes, average="micro")
        self.accuracy2 = MulticlassAccuracy(num_classes=t_classes, average="micro") #added, is the one of the training
        self.top5_accuracy = MulticlassAccuracy( #Likely to be out
            num_classes=n_classes, top_k=5, average="micro"
        )
        # 1: To print the accu
        self.train_loss = [] # Added, save outputs in each batch to compute metric overall epoch
        self.train_accu = [] # Added, save targets in each batch to compute metric overall epoch
        self.val_loss = [] # Added, save outputs in each batch to compute metric overall epoch
        self.val_accu = [] # Added, save targets in each batch to compute metric overall epoch
    
    def get_char(self,batch):
        print(batch,type(batch),batch.shape)
        print(batch[0],type(batch[0]),batch[0].shape)

    def forward(self, batch): #Edit
        x = self.spectrum_embedder(batch) #Where 'x' is the input
        a = self.seq_embedder(batch).reshape(-1, self.embed_dim) #And 'a' is the attribute or side information that work as identifier of the class
        #Get the similarity scores
        scores = torch.matmul(x, a.t()) #Check dim maybe need to transpose
        return scores #maybe need to softmax

    def training_step(self, batch, batch_idx): #Editing
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)
        batch['seq_ohe'].to(self.dtype) #Maybe?

        logits = self(batch) #The model is based on the MLP/CNNNemebdding, that specify on the foward that 'intensity' and 'seq' are the inputs so you can use the whole batch as input
        loss = F.cross_entropy(logits, batch["strain"])
        self.log("train_loss", loss, batch_size=len(batch["strain"])) 
        accu = self.accuracy2(logits, batch["strain"]) #Added
        #Consider a log for the accu

        self.log( #added
            "train_acc",
            self.accuracy2,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["strain"]),
        )

        self.train_loss.append(loss) #self.train_loss.extend(loss) #Added
        self.train_accu.append(accu) #self.train_accu.extend(self.accuracy2) #Added 

        return loss

    def validation_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        logits = self(batch)
        loss = F.cross_entropy(logits, batch["strain"]) #originally species, change to strain

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["strain"]),
        )

        accu = self.accuracy(logits, batch["strain"])
        self.log(
            "val_acc",
            self.accuracy,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["strain"]),
        )
        self.top5_accuracy(logits, batch["strain"])
        self.log(
            "val_top5_acc",
            self.top5_accuracy,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["strain"]),
        )

        #To print the accu and loss
        self.val_loss.append(loss) #.extend(loss) #Added
        self.val_accu.append(accu) #.extend(self.accuracy) #Added 

    def on_validation_epoch_end(self):
        # Aggregate results and print accuracy and loss
        if len(self.train_loss) == 0:
            epoch_train_loss = "NA"
            epoch_train_accu = 0.0  #self.train_accu.mean()            
        else:
            epoch_train_loss = sum(self.train_loss)/len(self.train_loss)  #self.train_loss.mean()
            epoch_train_accu = sum(self.train_accu)/len(self.train_accu)  #self.train_accu.mean()
        print(f"\nEpoch {self.current_epoch} - Train loss: {epoch_train_loss}, Train accu: {epoch_train_accu}")
        self.train_loss = [] #self.train_loss.clear()
        self.train_accu = [] #self.train_accu.clear()  
        
        # Aggregate results and print accuracy and loss
        epoch_val_loss = sum(self.val_loss)/len(self.val_loss) #self.val_loss.mean()
        epoch_val_accu = sum(self.val_accu)/len(self.val_accu) #self.val_accu.mean()
        print(f"Epoch {self.current_epoch} - Val loss: {epoch_val_loss}, Val accu: {epoch_val_accu}\n")
        self.val_loss = [] #.clear()
        self.val_accu = [] #.clear()  


    def predict_step(self, batch, batch_idx): #Here the data set will return a vector with the scores of all the analized seq in the seq minibatc
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)

        logits = self(batch)

        return (
            torch.stack([batch["strain"].to(logits).unsqueeze(-1), logits], -1),
            batch["0/loc"],
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lambd = lambda epoch: self.hparams.lr_decay_factor
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lambd
        )
        return [optimizer], [lr_scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        # warm up lr
        if self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams.warmup_steps
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def n_params(self):
        params_per_layer = [(name, p.numel()) for name, p in self.named_parameters()]
        total_params = sum(p.numel() for p in self.parameters())
        params_per_layer += [("total", total_params)]
        return params_per_layer
