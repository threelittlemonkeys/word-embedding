import sys
import re
from model import *
from utils import *

def load_model():
    vocab = load_vocab(sys.argv[2])
    model = cbow(len(vocab))
    if CUDA:
        model = model.cuda()
    print(model)
    load_checkpoint(sys.argv[1], model)
    return model, vocab

def run_model(model, data):
    batch = []
    z = len(data)
    while len(data) < BATCH_SIZE:
        data.append(["", UNK_IDX])
    for x in data:
        batch.append(x[1])
    result = model.embed(LongTensor(batch))
    for i in range(z):
        data[i].append(result[i].data)
    return data[:z]

def evaluate():
    k = 20
    word = sys.argv[3]
    data = []
    result = []
    model, vocab = load_model()
    if word in vocab:
        data.append([word, vocab[word]])
        del vocab[word]
    else:
        data.append([word, UNK_IDX])
    for word in vocab:
        data.append([word, vocab[word]])
        if len(data) == BATCH_SIZE:
            result.extend(run_model(model, data))
            data = []
    if len(data):
        result.extend(run_model(model, data))
    for x in result:
        x.append(torch.dist(x[2], result[0][2]))
    for x in sorted(result, key = lambda x: x[3], reverse = True)[:k]:
        print(x[0], x[3])

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage: %s model vocab word" % sys.argv[0])
    print("cuda: %s" % CUDA)
    evaluate()
