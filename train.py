import sys
import re
import time
from model import *
from utils import *
from os.path import isfile

def load_data():
    data = []
    batch_context = []
    batch_word = []
    print("loading data...")
    vocab = load_vocab(sys.argv[2])
    fo = open(sys.argv[3], "r")
    for line in fo:
        line = line.strip()
        context = [int(i) for i in line.split(" ")]
        word = context.pop()
        batch_context.append(context)
        batch_word.append([word])
        if len(batch_word) == BATCH_SIZE:
            data.append((Var(LongTensor(batch_context)), Var(LongTensor(batch_word))))
            batch_context = []
            batch_word = []
    fo.close()
    print("data size: %d" % (len(data) * BATCH_SIZE))
    print("batch size: %d" % BATCH_SIZE)
    return data, vocab

def train():
    num_epochs = int(sys.argv[4])
    data, vocab = load_data()
    model = cbow(len(vocab))
    optim = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    epoch = load_checkpoint(sys.argv[1], model) if isfile(sys.argv[1]) else 0
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    print(model)
    print("training model...")
    for i in range(epoch + 1, epoch + num_epochs + 1):
        loss_sum = 0
        timer = time.time()
        for x, y in data:
            loss = 0
            model.zero_grad()
            loss = F.nll_loss(model(x), y.squeeze(1))
            loss.backward()
            optim.step()
            loss = scalar(loss) / len(x)
            loss_sum += loss
        timer = time.time() - timer
        loss_sum /= len(data)
        if i % SAVE_EVERY and i != epoch + num_epochs:
            save_checkpoint("", "", i, loss_sum, timer)
        else:
            save_checkpoint(filename, model, i, loss_sum, timer)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s model vocab training_data num_epoch" % sys.argv[0])
    print("cuda: %s" % CUDA)
    train()
