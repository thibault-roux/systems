import numpy as np
import pickle
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def load_dict(system):
    ind2tok = dict()
    tok2ind = dict()
    ind = 0
    if "bpe" in system:
        type_tok = "bpe"
        number = system.split("bpe")[1].split("_")[0]
    else:
        type_tok = "char"
        number = "100"
    if number == "50":
        number = "70"
    with open("../results_lia_asr/" + system + "/1234/save/" + number + "_" + type_tok + ".vocab", "r") as f:
        for line in f:
            tok = line.split()[0]
            tok = tok.replace("▁", " ")
            ind2tok[ind] = tok
            tok2ind[tok] = ind
            ind += 1
    return ind2tok, tok2ind


def correlation(system1, system2): # vocab = {token, char, all}, normtype = {exp, norm}
    tensor1 = pickle.load(open("pickle/tensor_" + system1 + ".pkl", "rb"))
    tensor2 = pickle.load(open("pickle/tensor_" + system2 + ".pkl", "rb"))

    sentence = " alors nous avons un"

    # Load dict
    ind2tok1, tok2ind1 = load_dict(system1)
    ind2tok2, tok2ind2 = load_dict(system2)
    
    tensor1 = tensor1[:, :len(ind2tok1.keys())]
    tensor2 = tensor2[:, :len(ind2tok2.keys())]

    useful_toks = [' ', '<unk>', 'a', 'l', 'o', 'r', 's', 'n', 'u', 'v']

    # ranking
    indices1 = np.argsort(tensor1, axis=1)
    indices2 = np.argsort(tensor2, axis=1)
    tensor1 = np.argsort(indices1, axis=1)
    tensor2 = np.argsort(indices2, axis=1)

    # for i in range(tensor1.shape[0]):
    #     # select the indice of the lowest value of the numpy array
    #     indice = np.argmax(tensor1[i])
    #     print(i, indice, ind2tok1[indice])

    # example of correlation of a token between two systems
    # id1 = tok2ind1["a"]
    # id2 = tok2ind2["a"]
    # print(tensor1[:, id1]) # 65 signifie que c'est le token choisi
    # print(tensor2[:, id2])
    # # correlation between the rows
    # print(scipy.stats.spearmanr(tensor1[:, id1], tensor2[:, id2]))




    # calculer la corrélation entre deux systèmes {proches, différents} et entre deux tokens {même, autre}

    # calculer la corrélation entre deux systèmes
    # calculer correlation[system1, system2] pour tous les tokens
    correlation_matrix = np.zeros((len(useful_toks), len(useful_toks)))
    for tok1 in useful_toks:
        for tok2 in useful_toks:
            id1 = tok2ind1[tok1]
            id2 = tok2ind2[tok2]
            corr, pval = scipy.stats.spearmanr(tensor1[:, id1], tensor2[:, id2])
            correlation_matrix[useful_toks.index(tok1)][useful_toks.index(tok2)] = corr
            # print(tok1, tok2, int(corr*1000)/1000, int(pval*1000)/1000)

    # Créer la heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label="Score de Corrélation")
    plt.xticks(np.arange(len(useful_toks)), useful_toks, rotation=45)
    plt.yticks(np.arange(len(useful_toks)), useful_toks)
    plt.title("Heatmap des Scores de Corrélation entre les Étudiants")
    plt.show()

    plt.savefig("figures/corr/heatmap.png")

if __name__ == "__main__":
    system1 = "wav2vec2_ctc_fr_bpe1000_7k"
    system2 =  "wav2vec2_ctc_fr_bpe150_7k"
    # double boucle for every pair of systems
    correlation(system1, system2)
