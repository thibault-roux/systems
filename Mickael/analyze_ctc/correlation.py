import numpy as np
import pickle
import scipy.stats

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

    # input()




    # calculer la corrélation entre deux systèmes {proches, différents} et entre deux tokens {même, autre}

    # est-ce que ça sert ce qui est fait là ?
    # moyenne des corrélation entre les tokens à l'intérieur du système
    already_seen = set()
    for tok1 in useful_toks:
        for tok2 in useful_toks:
            # check if (tok1, tok2) and (tok2, tok1) have already been seen
            if (tok1, tok2) in already_seen or (tok2, tok1) in already_seen:
                continue
            else:
                already_seen.add((tok1, tok2))
            id1 = tok2ind1[tok1]
            id2 = tok2ind1[tok2]
            corr, pval = scipy.stats.spearmanr(tensor1[:, id1], tensor1[:, id2])
            print(tok1, tok2, int(corr*1000)/1000, int(pval*1000)/1000)

    input()

    id1 = tok2ind1["a"]
    id2 = tok2ind2["a"]
    print(tensor1[:, id1]) # 65 signifie que c'est le token choisi
    print(tensor2[:, id2])
    # correlation between the rows
    print(scipy.stats.spearmanr(tensor1[:, id1], tensor2[:, id2]))

    # compute a matrix of correlation between the rows and plot the matrix
    # matrix = np.zeros((tensor1.shape[1], tensor2.shape[1]))
    # for tok1 in useful_toks:
    #     for tok2 in useful_toks:
    #         id1 = tok2ind1[tok1]
    #         id2 = tok2ind2[tok2]
    #         matrix[id1, id2] = scipy.stats.spearmanr(tensor1[:, id1], tensor2[:, id2])[0]
    #         print(tok1, tok2, scipy.stats.spearmanr(tensor1[:, id1], tensor2[:, id2]))



if __name__ == "__main__":
    system1 = "wav2vec2_ctc_fr_bpe1000_7k"
    system2 =  "wav2vec2_ctc_fr_bpe150_7k"
    correlation(system1, system2)
