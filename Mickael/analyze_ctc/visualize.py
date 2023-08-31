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
    if number == 70:
        number = 50
    with open("../results_lia_asr/" + system + "/1234/save/" + number + "_" + type_tok + ".vocab", "r") as f:
        for line in f:
            tok = line.split()[0]
            tok = tok.replace("▁", " ")
            ind2tok[ind] = tok
            tok2ind[tok] = ind
            ind += 1
    return ind2tok, tok2ind


def plot_ctc_heatmap(system):
    tensor = pickle.load(open("pickle/tensor_" + system, "rb"))

    sentence = "alors nous avons un"

    # Load dict
    ind2tok, tok2ind = load_dict(system
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

    # normalization
    # tensor = tensor / np.linalg.norm(tensor, axis=1, keepdims=True)
    # min_values = np.min(tensor, axis=1)
    # max_values = np.max(tensor, axis=1)
    # tensor = (tensor - min_values[:, np.newaxis]) / (max_values[:, np.newaxis] - min_values[:, np.newaxis])

    # compute the exponential of the tensor
    tensor = np.exp(tensor)

    # keep only the rows with interesting tokens
    filtered_tensor = tensor[:, useful_rows_ids]

    # transpose to have temporality in x-axis
    filtered_tensor = np.transpose(filtered_tensor)
    # Create a heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(filtered_tensor, cmap='viridis', aspect='auto')
    # plt.title("CTC Layer Output Heatmap")
    # plt.xlabel("Token Index")
    # plt.ylabel("Speech Segment")
    plt.yticks(np.arange(len(useful_toks)), useful_toks)
    plt.colorbar(label="Value")
    plt.show()


    plt.savefig("figures/heatmap_" + system + ".png")

if __name__ == "__main__":
    systems = ["wav2vec2_ctc_fr_1k.pkl", "wav2vec2_ctc_fr_3k.pkl", "wav2vec2_ctc_fr_7k.pkl", "wav2vec2_ctc_fr_bpe1000_7k.pkl", "wav2vec2_ctc_fr_bpe100_7k.pkl", "wav2vec2_ctc_fr_bpe1500_7k.pkl", "wav2vec2_ctc_fr_bpe150_7k.pkl", "wav2vec2_ctc_fr_bpe250_7k.pkl", "wav2vec2_ctc_fr_bpe500_7k.pkl", "wav2vec2_ctc_fr_bpe50_7k.pkl", "wav2vec2_ctc_fr_bpe650_7k.pkl", "wav2vec2_ctc_fr_bpe750_7k.pkl", "wav2vec2_ctc_fr_bpe900_7k.pkl", "wav2vec2_ctc_fr_xlsr_53_french.pkl", "wav2vec2_ctc_fr_xlsr_53.pkl"] 
    for system in systems:
        plot_ctc_heatmap(system)
        break