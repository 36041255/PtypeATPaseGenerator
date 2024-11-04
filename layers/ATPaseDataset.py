import torch.utils.data as data
import torch
import pandas as pd
from torch.utils.data import DataLoader
import glob
import os
from chroma import Chroma, Protein
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

mp.set_start_method('spawn', force=True)

class ATPaseDataset(data.Dataset):
    def __init__(self, PDB_dir, excel_path, device='cpu',max_workers=10):
        self.device = device
        self.Proteins = []
        self.labels = torch.tensor([], device=device)
        self.kinds = {"E1":0, "E2":0, "E2P":0, "E1P":0, "E2Pi":0}
        self.ID = []
        self.len = 0
        df = pd.read_excel(excel_path)
        accession_column = df['Accession']

        # 使用ProcessPoolExecutor并行化任务
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(len(accession_column)):
                # 注意，这里我们只传递必要的信息给进程
                futures.append(executor.submit(process_protein, PDB_dir, accession_column[i], df.loc[i]["State"], device))
            
            # 等待所有任务完成
            for future in futures:
                result = future.result()
                if result is not None:
                    proteins, labels, state, accession_id = result                    
                    self.len += len(proteins)
                    self.kinds[state] += len(proteins)
                    for j in range(len(proteins)):
                        self.ID.append(accession_id)
                        self.Proteins.append(proteins[j])
                        self.labels = torch.cat((self.labels, labels[j].to(self.device)), dim=0)  # 注意转换设备

        print(self.kinds)

    def __getitem__(self, index):
        protein, label, ID = self.Proteins[index], self.labels[index],self.ID[index]
        return protein, label, ID

    def __len__(self):
        return self.len
    

def S2O(St):
    Ot = F.one_hot(St, 20).float()
    return Ot

def state2label(state, device):
    label = None
    _state = None
    if pd.isna(state):
        return label, _state

    if "E1" in state:
        if "E1P" in state or "E1-Pi" in state:  # E1P
            label = torch.tensor([[[0,1.0000,0,0]]], device=device)*1.15
            _state = "E1P"
        else:  # E1
            label = torch.tensor([[[1.0000,0,0,0]]], device=device)*1.15
            _state = "E1"
    elif "E2" in state:
        if "E2P" in state:  # E2P
            label = torch.tensor([[[0,0,1.0000,0]]], device=device)
            _state = "E2P"
#         elif "E2-Pi" in state:  # E2-Pi
#             label = torch.tensor([[[0,0,0.65000,0.3500]]], device=device)
#             _state = "E2Pi"
        elif "E2" in state and not "E2-Pi" in state:  # E2
            label = torch.tensor([[[0,0,0,1.0000]]], device=device)*1.1
            _state = "E2"
    return label, _state

def process_protein(PDB_dir, accession_id, state_str, device):
    label, state = state2label(state_str, device)
    if label is None:
        return None

    PDBfiles = glob.glob(os.path.join(PDB_dir, accession_id + '*.pdb'))
    proteins = []
    labels = torch.tensor([], device=device)
    print(accession_id,state,len(PDBfiles))
    for PDBfile in PDBfiles:
        protein = list(Protein(PDBfile, device=device).to_XCS())
        protein[2] = S2O(protein[2])
        proteins.append(protein)
        labels = torch.cat((labels, label), dim=0)
    
    return proteins, labels, state, accession_id


def label2state(label):
    maxarg = torch.argmax(label)
    if maxarg == 0:
        return "E1"
    elif maxarg == 1:
        return "E1P"
    elif maxarg == 2:
        return "E2P"
    elif maxarg == 3:
        return "E2"
