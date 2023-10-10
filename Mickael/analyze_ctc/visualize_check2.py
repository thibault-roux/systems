import numpy as np
import pickle
import matplotlib.pyplot as plt


def load_dict(system):
    ind2tok = dict()
    tok2ind = dict()
    ind = 0
    with open("../results_lia_asr/wav2vec2_ctc_fr_bpe900_CHECK_7k/1234/save/900_bpe.vocab", "r") as f:
        for line in f:
            tok = line.split()[0]
            tok = tok.replace("▁", " ")
            ind2tok[ind] = tok
            tok2ind[tok] = ind
            ind += 1
    return ind2tok, tok2ind


def get_lengths(system): # vocab = {token, char, all}, normtype = {exp, norm}
    tensor = pickle.load(open("pickle3/tensor_" + system + ".pkl", "rb"))[0]

    sentence = " alors nous avons un"
    # sentence = " à la mémoire d' un général battu monsieur bruno le maire"

    # Load dict
    ind2tok, tok2ind = load_dict(system)
    
    length = []
    for i in range(tensor.shape[0]):
        # select the indice of the highest values of the numpy array that correspond to a token of length 1
        indices = np.argsort(tensor[i])
        indices = indices[::-1] # reverse the array
        for indice in indices:
            # if len(ind2tok[indice]) == 1:
            if indice in ind2tok.keys() and ind2tok[indice] != "<unk>": # and ind2tok[indice] in useful_toks:
                # print(i, ind2tok[indice], end="\n")
                length.append(len(ind2tok[indice]))
                break

    return length

if __name__ == "__main__":
    total_length = []

    systems = [str(i) for i in range(36000, 36050, 50)]
    for system in systems:
        print(system)
        total_lenght.extend(get_lengths(system)) # vocab = {token, char, all}, normtype = {exp, norm}