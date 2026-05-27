import os
import torch
import numpy as np

from src.utils.train_test import evaluate
from src.utils.GoogleSpeechCommands import SubsetSC

import matplotlib.pyplot as plt


device = 'cpu'
pin_memory = False
seed = 1234
num_workers = 0
batch_size = 1


device = torch.device(device if torch.cuda.is_available() else "cpu")

# set seed for pytorch and numpy
torch.manual_seed(seed)
np.random.seed(seed)

# Load data
print("Start loading data")
dataset_path = "./data/SpeechCommands/"

test_set = SubsetSC(dataset_path, "testing")
print("End loading data")

print("Generate Datalaoder")
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)

for inputs, labels in test_loader:
    print(inputs.shape)
    break

# test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
print("Generate Datalaoder end")


# load model
import CppSEdge

CppSEdge.printModelInfo()
print("CppSEdge loaded")

print("Desired Input Size:", CppSEdge.getInputSize())

class model_wrapper:
    def __init__(self) -> None:
        self.device = torch.device('cpu')
        pass
    def eval(self):
        pass
    def __call__(self, input):
        # S5_output,classes = CppS5.run_S5(input.cpu().numpy().astype(np.float32))

        classes = CppSEdge.run(input.cpu().numpy().astype(np.float32))
        classes = torch.tensor(classes).to(input.device)
        return classes

transform = None  # transform.to(device)

criterion = torch.nn.CrossEntropyLoss()

print('Evaluate on test set')

test_loss, test_acc, confusion_matrix = evaluate(model_wrapper(), criterion, test_loader, transform=transform, return_confusion=True)
print(f'Test loss: {test_loss}, Test accuracy: {test_acc}')
print(CppSEdge.getMetaInfo())

confusion_matrix = confusion_matrix
fig = plt.figure(figsize=(8, 8))
plt.imshow(confusion_matrix, cmap='hot', interpolation='nearest')
plt.xticks(np.arange(0, 35, 1),test_set.named_labels, rotation=-90)
plt.yticks(np.arange(0, 35, 1),test_set.named_labels)
# set range of colorbar
plt.clim(0, 100)
plt.colorbar()
plt.savefig(os.path.join('./', f'confusion_matrix_CPP.png'))
plt.close()

