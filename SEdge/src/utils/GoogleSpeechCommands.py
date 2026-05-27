import os
from torch.utils.data import Dataset
import torchaudio
import torch
from pathlib import Path
from tqdm import tqdm

try:
    n_cols = os.get_terminal_size().columns
except Exception as e:
    n_cols = 150
    print(f"Error getting terminal size: {e}. Defaulting to {n_cols} columns.")
    pass

####### DATALOADER #######

class SubsetSC(Dataset):
    def __init__(self, source_folder, subset: str = None, download: bool = True, device='cpu'):

        path = os.path.join('/dev/shm', os.environ['USER'], 'SpeechCommands/')


        HASH_DIVIDER = "_nohash_"
        EXCEPT_FOLDER = "_background_noise_"
 
        labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 
                  'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 
                  'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 
                  'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
        
        self.named_labels = labels 

        if not os.path.exists(source_folder):
            os.makedirs(source_folder)

        if download:
            torchaudio.datasets.SPEECHCOMMANDS(source_folder, download=download)

        source_folder_expanded = source_folder + "SpeechCommands/speech_commands_v0.02/"
 
        def pad_sequence(batch):
            # Make all tensor in a batch the same length by padding with zeros
            batch = [item.t() for item in batch]
            batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
            return batch
    
        def load_list(root, *filenames):
            output = []
            for filename in filenames:
                filepath = os.path.join(root, filename)
                with open(filepath) as fileobj:
                    output += [os.path.normpath(os.path.join(root, line.strip())) for line in fileobj]
            return output

        if subset == "validation":
            self.walker = load_list(source_folder_expanded, "validation_list.txt")
        elif subset == "testing":
            self.walker = load_list(source_folder_expanded, "testing_list.txt")
        elif subset == "training":

            self.walker = sorted(str(p) for p in Path(source_folder_expanded).glob("*/*.wav"))
            self.walker = [w for w in self.walker if HASH_DIVIDER in w and EXCEPT_FOLDER not in w]
            excludes = set(load_list(source_folder_expanded, "validation_list.txt", "testing_list.txt"))
            excludes = set(excludes)
            self.walker = [w for w in self.walker if w not in excludes]
        
        else:
            raise ValueError("subset should be 'training', 'validation' or 'testing'")

        # chache the wav files and the labels
        self.audios = []
        self.label_str = []
        self.labels = []
        if not os.path.exists(path):
            os.makedirs(path)
        if os.path.exists(path + subset + 'audios.pt') and os.path.exists(path + subset + 'labels.pt'):
            self.audios = torch.load(path + subset + 'audios.pt',weights_only=True)
            self.labels = torch.load(path + subset + 'labels.pt',weights_only=True)
        else:
            for item in tqdm(self.walker, desc=f"IGNORE: Loading {subset} set", ncols=n_cols):
                waveform, sample_rate = torchaudio.load(item)
                self.audios.append(waveform)
                label_str = os.path.basename(os.path.dirname(item))
                label = labels.index(label_str)
                self.labels.append(label)

            self.sample_rate = sample_rate
            self.audios = pad_sequence(self.audios)
            self.labels = torch.tensor(self.labels)
            
            if subset != "training":
                if not os.path.exists(path + "mean.pt") or not os.path.exists(path + "std.pt"):
                    SubsetSC(source_folder, "training", download=download, device=device)
                self.mean = torch.load(path + "mean.pt")
                self.std = torch.load(path + "std.pt")
            else:
                #assumse single input channel
                self.mean = self.audios.mean()
                self.std = self.audios.std()
                torch.save(self.mean, path + "mean.pt")
                torch.save(self.std, path + "std.pt")
            
            print(f"Dataset mean={self.mean} std={self.std}")
            self.audios = (self.audios - self.mean) / (self.std+1e-5)
            
            torch.save(self.audios, path + subset + 'audios.pt')
            torch.save(self.labels, path + subset + 'labels.pt')
        
        self.audios = self.audios.to(device)
        self.labels = self.labels.to(device)

        self.device = device

    def __len__(self) -> int:
        return len(self.walker)


    def get(self, idx):
        return self.audios[idx,...], self.labels[idx,...]
        pass

    def __getitem__(self, idx):
        if self.device != 'cpu':
            return idx
        # return idx
        audio = self.audios[idx]
        label = self.labels[idx]

        return audio, label

if __name__ == '__main__':
    dataset_path = "./data/SpeechCommands/"

    train_set = SubsetSC(dataset_path, "training")
    test_set = SubsetSC(dataset_path, "testing")
    valid_set = SubsetSC(dataset_path, "validation")
