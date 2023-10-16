import pandas as pd
import matplotlib.pyplot as plt


def read_file(filename):
    with open(filename, "r") as f:
        vocab = f.readlines()
        vocab = [line.split()[0] for line in vocab]
    return vocab

# check if tokens in vocabulary for a BPE of a certain size are included in the vocabulary of a BPE of a larger size
def previous_main():
    # systems = ["50", "100", "150", "250", "500", "650", "750", "900", "1000", "1500"]
    systems = ["150", "250", "500", "750", "1000"]
    vocab2 = []
    for system in systems:
        print("System processed: " + system)
        if system == "50":
            system2 = "70"
        else:
            system2 = system
        filename = "results_lia_asr/wav2vec2_ctc_fr_bpe" + system + "_7k_without_space/1234/save/" + system2 + "_bpe.vocab"
        vocab = read_file(filename)
        if len(vocab2) == 0:
            vocab2 = vocab
        else:
            absent_tokens = []
            for token in vocab:
                if token not in vocab2:
                    absent_tokens.append(token)
            print(absent_tokens)
            print("Number of tokens that are in " + system + " but not in the previous: " + str(len(absent_tokens)))
            vocab2 = vocab
        input()



# plot histograms of token lengths for different BPE sizes
if __name__ == "__main__":
    data = dict()
    systems = ["250", "500", "750", "1000"]
    for system in systems:
        filename = "results_lia_asr/wav2vec2_ctc_fr_bpe" + system + "_7k/1234/save/" + system + "_bpe.vocab"
        vocab = read_file(filename)
        token_lengths = [len(token) for token in vocab]
        data[system] = pd.DataFrame({'Token Length': token_lengths})

    # Create subplots for each vocabulary
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
    fig.suptitle('Token Length Distribution for Different Vocabularies')

    # Plot histograms for each vocabulary
    for i, (df, ax) in enumerate(zip([data["250"], data["500"], data["750"], data["1000"]], axes.flat)):
        df['Token Length'].plot(kind='hist', ax=ax, bins=10)
        ax.set_title(f'Vocabulary {i+1}')
        ax.set_xlabel('Length of tokens')
        ax.set_ylabel('Frequency')

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Show the plot
    plt.show()
    plt.savefig("hist_token_length.png")
