import sys
from model import VOCAB_SIZE, CONTEXT_SIZE, UNK, UNK_IDX
from utils import tokenize

def load_data():
    data = []
    freq = {}
    vocab = {UNK: UNK_IDX}
    fo = open(sys.argv[1])
    for line in fo:
        tokens = tokenize(line)
        for word in tokens:
            if word not in freq:
                freq[word] = 0
            freq[word] += 1
    for word, _ in sorted(freq.items(), key = lambda x: x[1], reverse = True):
        vocab[word] = len(vocab)
        if len(vocab) == VOCAB_SIZE:
            break
    fo.seek(0)
    for line in fo:
        tokens = tokenize(line)
        if len(tokens) < CONTEXT_SIZE * 2 + 1:
            continue
        seq = []
        for word in tokens:
            seq.append(vocab[word] if word in vocab else UNK_IDX)
        for i in range(CONTEXT_SIZE, len(seq) - CONTEXT_SIZE):
            context = seq[i - CONTEXT_SIZE:i + CONTEXT_SIZE + 1]
            if any(x == UNK_IDX for x in context):
                continue
            del context[CONTEXT_SIZE]
            data.append(context + [seq[i]])
    fo.close()
    return data, vocab

def save_data(data):
    fo = open(sys.argv[1] + ".csv", "w")
    for seq in data:
        fo.write(" ".join([str(i) for i in seq]) + "\n")
    fo.close()

def save_vocab(vocab):
    fo = open(sys.argv[1] + ".vocab", "w")
    for word, _ in sorted(vocab.items(), key = lambda x: x[1]):
        fo.write(word + "\n")
    fo.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    data, vocab = load_data()
    save_data(data)
    save_vocab(vocab)
