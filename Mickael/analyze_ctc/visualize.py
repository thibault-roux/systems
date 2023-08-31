import numpy as np
import pickle
import matplotlib.pyplot as plt


def load_dict():
    ind2tok = dict()
    tok2ind = dict()
    ind = 0
    with open("../results_lia_asr/wav2vec2_ctc_fr_test2/1234/save/750_bpe.vocab", "r") as f:
    # with open("../results_lia_asr/wav2vec2_ctc_fr_test2_char/1234/save/100_char.vocab", "r") as f:
        for line in f:
            tok = line.split()[0]
            tok = tok.replace("▁", " ")
            ind2tok[ind] = tok
            tok2ind[tok] = ind
            ind += 1
    return ind2tok, tok2ind


if __name__ == "__main__":
    # p_ctc = pickle.load(open("p_ctc_char.pkl", "rb"))[0].cpu()
    # tensor = p_ctc.numpy()
    # pickle.dump(tensor, open("tensor_char.pkl", "wb"))

    # tensor = pickle.load(open("tensor_char.pkl", "rb"))
    tensor = pickle.load(open("tensor.pkl", "rb"))

    sentence = "alors nous avons un"
    # print(sentence)

    # Load dict
    ind2tok, tok2ind = load_dict()
    useful_toks = set()
    useful_toks.add("<unk>")
    for ind, tok in ind2tok.items():
        if tok in sentence:
            useful_toks.add(tok)
    useful_toks = list(useful_toks)
    # useful_toks = [' ', ' a', '<unk>', 'a', 'al', ' alors', 'l', 'lo', 'o', 'or', 'ors', 'r', 's']
    useful_toks = [' ', '<unk>', 'a', 'l', 'o', 'r', 's', 'n', 'u', 'v']
    useful_toks.sort()
    useful_rows_ids = [tok2ind[tok] for tok in useful_toks]
    print(useful_toks)
    
    # print(useful_toks)
    # print(useful_rows_ids)

    # filtered_tensor = tensor

    # # print the id corresponding to the highest value of the tensor
    # useful_columns_ids = []
    # for i in range(filtered_tensor.shape[0]):
    #     print(i, np.argmax(filtered_tensor[i]))
        # if np.argmax(filtered_tensor[i]) != 0:
        #    useful_columns_ids.append(i)


    # tensor = tensor / np.linalg.norm(tensor, axis=1, keepdims=True)
    # min_values = np.min(tensor, axis=1)
    # max_values = np.max(tensor, axis=1)
    # tensor = (tensor - min_values[:, np.newaxis]) / (max_values[:, np.newaxis] - min_values[:, np.newaxis])

    # compute the exponential of the tensor
    tensor = np.exp(tensor)


    # old visualisation
    # for i in range(tensor.shape[0]):
    #     indice = np.argmax(tensor[i])
    #     # print(np.sum(tensor[i]))
    #     print(i, ind2tok[indice], indice, tensor[i][indice])
    # input()

    # keep only the rows with interesting tokens
    filtered_tensor = tensor[:, useful_rows_ids]
    # filtered_tensor = filtered_tensor[useful_columns_ids, :]

    # transpose to have temporality in x-axis
    filtered_tensor = np.transpose(filtered_tensor)
    
    # Create a heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(filtered_tensor, cmap='viridis', aspect='auto')

    # Customize the plot as needed
    # plt.title("CTC Layer Output Heatmap")
    # plt.xlabel("Token Index")
    # plt.ylabel("Speech Segment")
    plt.yticks(np.arange(len(useful_toks)), useful_toks)
    plt.colorbar(label="Value")

    # Show the plot
    plt.show()


    # plt.savefig("heatmap_char.png")
    plt.savefig("heatmap2_temp.png")
