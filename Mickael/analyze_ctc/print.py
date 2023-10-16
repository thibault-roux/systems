import numpy as np
import pickle
import matplotlib.pyplot as plt


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


def print_ctc(system):
    tensor = pickle.load(open("pickle/tensor_" + system + ".pkl", "rb"))

    sentence = " alors nous avons un"
    # sentence = " à la mémoire d' un général battu monsieur bruno le maire"

    # Load dict
    ind2tok, tok2ind = load_dict(system)
    
    # useful_toks = [' ', '<unk>', 'a', 'l', 'o', 'r', 's', 'n', 'u', 'v']
    useful_toks = list(set(sentence))
    # useful_toks.append("<unk>")

    print(tensor.shape)

    for i in range(tensor.shape[0]):
        # select the indice of the highest values of the numpy array that correspond to a token of length 1
        indices = np.argsort(tensor[i])
        indices = indices[::-1] # reverse the array
        for indice in indices:
            if len(ind2tok[indice]) == 1:
                print(ind2tok[indice], end=" ")
                break
    print()

if __name__ == "__main__":
    # systems = ["wav2vec2_ctc_fr_1k", "wav2vec2_ctc_fr_3k", "wav2vec2_ctc_fr_7k", "wav2vec2_ctc_fr_bpe1000_7k", "wav2vec2_ctc_fr_bpe100_7k", "wav2vec2_ctc_fr_bpe1500_7k", "wav2vec2_ctc_fr_bpe150_7k", "wav2vec2_ctc_fr_bpe250_7k", "wav2vec2_ctc_fr_bpe500_7k", "wav2vec2_ctc_fr_bpe50_7k", "wav2vec2_ctc_fr_bpe650_7k", "wav2vec2_ctc_fr_bpe750_7k", "wav2vec2_ctc_fr_bpe900_7k", "wav2vec2_ctc_fr_xlsr_53_french", "wav2vec2_ctc_fr_xlsr_53"] 
    systems = ["wav2vec2_ctc_fr_7k", "wav2vec2_ctc_fr_bpe1000_7k"]
    for system in systems:
        print(system)
        print_ctc(system)