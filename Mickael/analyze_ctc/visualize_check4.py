import numpy as np
import pickle
import matplotlib.pyplot as plt


def load_dict():
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


def compute_matrix(system, ind2tok, tok2ind, X=None):
    tensor = pickle.load(open("pickle3/tensor_" + system + ".pkl", "rb"))[0]

    sentence = " alors nous avons un"
    # sentence = " à la mémoire d' un général battu monsieur bruno le maire"

    
    
    # print(tensor.shape) # (42, 900), there is 42 segments and 900 tokens in the vocabulary
    # I want the rank for each segment
    tensor_rank = np.zeros((tensor.shape[0], tensor.shape[1]))
    for i in range(tensor.shape[0]):
        tensor_rank[i] = np.argsort(tensor[i])[::-1] # reverse the array

    # si X non null
    if X is None:
        return tensor_rank
    else:
        return [tensor_rank[X]]

    # ici, je veux faire la heatmap sur les rangs de chaque token pour voir l'évolution au cours de l'entraînement
    # je peux soit plot la moyenne des segments, soit les afficher à la suite, soit choisir un segment au hasard

if __name__ == "__main__":
    # Load dict
    ind2tok, tok2ind = load_dict()


    systems = [str(i) for i in range(0, 3000, 1)] # (0, 38000, 50)]
    # systems = ["wav2vec2_ctc_fr_7k", "wav2vec2_ctc_fr_bpe1500_7k"]
    matrix = np.array([])
    for system in systems:
        print(system)
        map2 = compute_matrix(system, ind2tok, tok2ind, X=10)
        try:
            matrix = np.concatenate((matrix, map2), axis=0)
        except ValueError:
            matrix = map2


    token_lengths = [len(ind2tok[x]) for x in range(matrix.shape[1])]
    sorted_indices = np.argsort(token_lengths)
    sorted_matrix = matrix[:, sorted_indices] # sort

    sorted_matrix = sorted_matrix.T
    print(sorted_matrix.shape)
    # Create the heatmap
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    plt.imshow(sorted_matrix, cmap='viridis', aspect='auto')
    plt.colorbar()

    counter = dict()
    for i in range(len(token_lengths)):
        try:
            counter[token_lengths[i]] += 1
        except KeyError:
            counter[token_lengths[i]] = 1

    # Add horizontal grid lines to separate tokens
    previous = 0
    for token_length in range(1, max(counter.keys())):
        count = counter[token_length]
        previous = previous + count
        plt.axhline(y=previous, color='red', linewidth=2)

    # Set axis labels
    plt.xlabel('Training')
    plt.ylabel('Tokens')

    # Show the plot
    plt.title('Heatmap of Probability Matrix')
    plt.show()
    plt.savefig("results/rank_heatmap.png")
