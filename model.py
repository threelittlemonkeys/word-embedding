import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

BATCH_SIZE = 64
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
        self.linear1 = nn.Linear(EMBED_SIZE, EMBED_SIZE // 2)
        self.linear2 = nn.Linear(EMBED_SIZE // 2, vocab_size)

        if CUDA:
            self = self.cuda()

    def forward(self, x):
        h = self.embed(x)
        h = h.sum(1)
        h = self.linear1(h)
        h = F.relu(h)
        h = self.linear2(h)
        y = F.log_softmax(h, 1)
        return y

def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if CUDA else x

def scalar(x):
    return x.view(-1).data.tolist()[0]
