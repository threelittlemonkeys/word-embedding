import re
from model import *

def tokenize(s):
    s = s.lower()
    s = re.sub("[^ a-z0-9\uAC00-\uD7A3]+", "", s)
    s = re.sub("\s+", " ", s)
    s = re.sub("^ | $", "", s)
    return s.split(" ")

def load_vocab(filename):
    print("loading vocab...")
    vocab = {}
    fo = open(filename)
    for line in fo:
        line = line.strip()
        vocab[line] = len(vocab)
    fo.close()
    return vocab

def load_checkpoint(filename, model = None, g2c = False):
    print("loading model...")
    if g2c: # load weights into CPU
        checkpoint = torch.load(filename, map_location = lambda storage, loc: storage)
    else:
        checkpoint = torch.load(filename)
    if model:
        model.load_state_dict(checkpoint["state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("saved model: epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))
    return epoch

def save_checkpoint(filename, model, epoch, loss):
    print("saving model...")
    checkpoint = {}
    checkpoint["state_dict"] = model.state_dict()
    checkpoint["epoch"] = epoch
    checkpoint["loss"] = loss
    torch.save(checkpoint, filename + ".epoch%d" % epoch)
    print("saved model: epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))

def gpu2cpu(filename):
    checkpoint = torch.load(filename, map_location = lambda storage, loc: storage)
    torch.save(checkpoint, filename + ".cpu")
