import numpy as np
import torch
from torch.utils.data import Dataset
import os
import torchaudio
  
class ASVSpoof2019LA(Dataset):
    def __init__(self, path_to_features, path_to_data, part='train'):
        super(ASVSpoof2019LA, self).__init__()
        self.path_to_features = os.path.join(path_to_features, part)
        self.path_to_data = os.path.join(path_to_data, part)
        self.part = part
        
        labels = open(os.path.join(self.path_to_data, "text")).read().splitlines()
        labels = [x for x in labels if x != ""]
        
        
        # match wavscp with labels
        self.files = []
        for i in range(len(labels)):
            uttID = labels[i].split()[0]
            label = int(labels[i].split()[1])
            attackType = labels[i].split()[2]
            self.files.append((uttID, label, attackType))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        uttID = self.files[idx][0]
        label = self.files[idx][1]
        attackType = self.files[idx][2]
        featureTensor = torch.load(os.path.join(self.path_to_features, uttID + ".pt"))
        return featureTensor.squeeze(0), label, attackType

if __name__ == "__main__":
    dataset = ASVSpoof2019LA(part="train")
    