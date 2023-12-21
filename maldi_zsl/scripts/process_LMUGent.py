import sys
from Bio import SeqIO
import numpy as np
import re
import lpsn
from tqdm import tqdm
import requests
from Bio import Entrez
import shutil
from maldi_nn.spectrum import *
import h5torch
import os


SILVA_path = str(sys.argv[0])
labelsshortdb_path = str(sys.argv[1])
spectradb_path = str(sys.argv[2])
lpsn_email = str(sys.argv[3])
lpsn_pw = str(sys.argv[4])
h5torch_raw = str(sys.argv[5])
h5torch_binned = str(sys.argv[6])
alignment_file = str(sys.argv[7])

########################################################
# 1 READ IN ALL SILVA SEQUENCES
########################################################
fasta_sequences = list(SeqIO.parse(open(SILVA_path),'fasta'))

########################################################
# 2 READ IN ALL LABELS
########################################################

lines = []
with open(labelsshortdb_path) as f:
    for line in f:
        lines.append(line.rstrip("\n"))

uniq_strains = np.unique([l.removeprefix("root;") for l in lines if len(l.split(";")) == 5])
uniq_strains = np.array([u.split(";") for u in uniq_strains])

########################################################
# 3 MATCH STRAIN NAMES TO SEQUENCES, MORE INFO SEE README
########################################################

aas = []
ncbi_ids = []

Entrez.email = 'Entrez@example.com'
client = lpsn.LpsnClient(lpsn_email, lpsn_pw) # Add your login details from lpsn bacterio net here.

def lookup(str_): # Straininfo lookup helper function
    req = requests.get('https://api.straininfo.dsmz.de/v1/search/culture/str_des/%s' % (str_)).json()
    return requests.get('https://api.straininfo.dsmz.de/v1/data/culture/max/%s' % (req[0])).json()


for ix in tqdm(range(len(uniq_strains))): # iterate over unique strains in detaset
    key = uniq_strains[ix, -1].replace(" ", "%20")
    try:
        res = lookup(key) # look up strain in straininfo
    except:
        # If this fails:
        # Some strains are called e.g. Genus species DSM xxx (B ...), this repeats the search without the (B ...)
        key = re.sub("\(.*\)", "", uniq_strains[ix, -1]).rstrip().replace(" ", "%20") 
        try:
            res = lookup(key)
        except:
            aas.append("Culture not found")
            continue
    
    # get the strain names from the matched culture in straininfo
    match_list = [res[0]["culture"]["strain_number"]]
    if "relation" in res[0]["culture"]:
        match_list += res[0]["culture"]["relation"]
    
    # get the genus and species name from the matched culture in straininfo
    try:
        taxon = res[0]["culture"]["taxon"]["name"]
    except:
        aas.append("No culture id in found record")
        continue

    # only if the genus, species and culture name correspond to the one we gave (i.e. no false match), proceed
    if any([m in key.replace("%20", " ") for m in match_list]) and any([l in uniq_strains[ix] for l in taxon.split(" ")]):
        # try to look up sequences in the straininfo match
        try:
            for l in sorted(res[0]["strain"]["sequence"], key= lambda x:-x["year"]):
                aa = [str(t.seq) for t in fasta_sequences if l["accession_number"] in t.description]
                if len(aa) > 0:
                    break
            if len(aa) == 0:
                raise ValueError
            aas.append(aa)
            ncbi_ids.append(res[0]["culture"]["taxon"])
            
        except:
            # if no sequence in the straininfo match, look up the strain in LPSN database.
            try:
                count = client.search(id=res[0]["culture"]["taxon"]["lpsn"])
                if count > 0:
                    entry = list(client.retrieve())[0]
                    if any([m in key.replace("%20", " ") for m in entry["type_strain_names"]]):
                        for m in entry["molecules"]:
                            if m["kind"] == "16S rRNA gene":
                                aa = [str(t.seq) for t in fasta_sequences if m["identifier"] in t.description]
                                if len(aa) > 0:
                                    break
                        if len(aa) == 0:
                            raise ValueError
                        aas.append(aa)
                        ncbi_ids.append(res[0]["culture"]["taxon"])
                    else:
                        aas.append("No matching seq found in lpsn")
                else:
                    aas.append("No matching seq found in lpsn")
            except:
                aas.append("No matching seq found")
    else:
        aas.append("Culture name found but matched to wrong ID in straininfo.")


name_to_seq = {}
for i in range(len(uniq_strains)):
    if len(aas[i][0]) > 1:
        name_to_seq[";".join(uniq_strains[i])] = aas[i][0]
        
name_to_ix = {name : ix for ix, name in enumerate(name_to_seq)}

########################################################
# 4 FOR ALL STRAINS THAT ARE MATCHED, MAKE A H5TORCH FILE
########################################################

spectra = []
species_ix = []

lines = []
f1 = open(labelsshortdb_path)
f2 = open(spectradb_path)

c = 0
for line1, line2 in zip(f1, f2):
    name = line1.removeprefix("root;").rstrip("\n")
    if name in name_to_seq:
        species_ix.append(name_to_ix[name])
        k = pd.Series(line2.rstrip().split(";")).str.split(":", expand=True)
        s = SpectrumObject(mz=k[0].astype(float), intensity=k[1].astype(int))
        spectra.append(s)

ix_to_name = {v : k for k, v in name_to_ix.items()}

f = h5torch.File(h5torch_raw, "w")

score_matrix = np.zeros((len(species_ix), len(name_to_ix)), dtype=bool)
score_matrix[np.arange(len(score_matrix)), species_ix] = 1
f.register(score_matrix, axis = "central", mode = "N-D", dtype_save = "bool", dtype_load = "int64")

ints = [s.intensity for s in spectra]
mzs = [s.mz for s in spectra]
f.register(ints, 0, name="intensity", mode="vlen", dtype_save = "int32", dtype_load="int64")
f.register(mzs, 0, name="mz", mode="vlen", dtype_save = "float32", dtype_load="float32")

f.register(np.array(strain_names), 1, name="strain_names", mode="N-D", dtype_save="bytes", dtype_load="str")
f.register(np.array(strain_seq), 1, name="strain_seq", mode="N-D", dtype_save="bytes", dtype_load="str")
f.close()

########################################################
# 5 MAKE A SECOND H5TORCH FILE, NOW WITH PREPROCESSED AND BINNED SPECTRA
########################################################

binner = SequentialPreprocessor(
    VarStabilizer(method="sqrt"),
    Smoother(halfwindow=10),
    BaselineCorrecter(method="SNIP", snip_n_iter=20),
    Trimmer(),
    Binner(),
    Normalizer(sum=1),
)
shutil.copy(h5torch_raw, h5torch_binned)
file = h5torch.File(h5torch_binned, "a")
len_ = file["0/mz"].shape[0]
ints = []
for i in tqdm(range(len_)):
    mz = file["0/mz"][i]
    intensity = file["0/intensity"][i]
    s = SpectrumObject(mz=mz, intensity=intensity)
    ints.append(binner(s).intensity)

del file["0/mz"]
del file["0/intensity"]

file.register(np.stack(ints), 0, name="intensity", dtype_save = "float32", dtype_load="float32")
file.register(binner(s).mz, "unstructured", name="mz", dtype_save = "float32", dtype_load="float32")

########################################################
# 6 MAKE AN ALIGNMENT, ENCODE TO INTEGERS, AND ADD TO H5TORCH FILE
########################################################

strain_seq = file["1/strain_seq"][:]
with open("tmp.fasta", "w") as f:
    for ix, seq in enumerate(strain_seq):
        f.write(">%s\n" % (ix))
        f.write("%s\n" % (seq))

os.system("mafft tmp.fasta > %s" % (alignment_file))
os.remove("tmp.fasta")
n = len(file["1/strain_seq"])

aligned_fasta_sequences = list(SeqIO.parse(open(alignment_file),'fasta'))
aligned_fasta_sequences = sorted(aligned_fasta_sequences, key = lambda x : int(x.id))

nucleotide_mapping = {"-" : 0, "a" : 1, "u": 2, "c" : 3, "g" : 4, "other" : 5}
sequences_encoded = []
for a in aligned_fasta_sequences:
    seq = str(a.seq)
    seq_encoded = np.array([(nucleotide_mapping[s] if s in nucleotide_mapping else 5) for s in seq], dtype="int8")
    sequences_encoded.append(seq_encoded)

file.register(np.array(sequences_encoded), axis = 1, name = "strain_seq_aligned", mode = "N-D", dtype_save="int8", dtype_load="int64")

