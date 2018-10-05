import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 128
EMBED_SIZE = 300
VOCAB_SIZE = 50000
CONTEXT_SIZE = 2
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
VERBOSE = True
SAVE_EVERY = 10

UNK = "<UNK>"
UNK_IDX = 0

torch.manual_seed(1)
CUDA = torch.cuda.is_available()

class cbow(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        # architecture
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE)
        self.fc1 = nn.Linear(EMBED_SIZE, EMBED_SIZE // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(EMBED_SIZE // 2, vocab_size)
        self.softmax = nn.LogSoftmax(1)

    def forward(self, x):
        h = self.embed(x)
        h = h.sum(1)
        h = self.fc1(h)
        h = self.relu(h)
        h = self.fc2(h)
        y = self.softmax(h)
        return y

class skipgram(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return

def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if CUDA else x

def scalar(x):
    return x.view(-1).data.tolist()[0]
