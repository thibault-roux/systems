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


def plot_ctc_heatmap(system):
    tensor = pickle.load(open("pickle3/tensor_" + system + ".pkl", "rb"))[0]

    sentence = " alors nous avons un"
    # sentence = " à la mémoire d' un général battu monsieur bruno le maire"

    # Load dict
    ind2tok, tok2ind = load_dict(system)
    
    # print(tensor.shape) # (42, 900), there is 42 segments and 900 tokens in the vocabulary
    # I want the rank for each segment
    tensor_rank = np.zeros((tensor.shape[0], tensor.shape[1]))
    for i in range(tensor.shape[0]):
        tensor_rank[i] = np.argsort(tensor[i])[::-1] # reverse the array

    return tensor_rank

    # ici, je veux faire la heatmap sur les rangs de chaque token pour voir l'évolution au cours de l'entraînement
    # je peux soit plot la moyenne des segments, soit les afficher à la suite, soit choisir un segment au hasard

if __name__ == "__main__":
    systems = [str(i) for i in range(7950, 38000, 50)]
    # systems = ["wav2vec2_ctc_fr_7k", "wav2vec2_ctc_fr_bpe1500_7k"]
    matrix = np.array([])
    for system in systems:
        print(system)
        matrix = np.append(matrix, plot_ctc_heatmap(system))
        print(matrix.shape)
        if system == systems[1]:
            exit(-1)