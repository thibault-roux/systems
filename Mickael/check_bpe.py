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




if __name__ == "__main__":
    systems = ["150", "250", "500", "750", "1000"]
    # Your lists of token lengths
    list1 = [3, 4, 4, 2, ...]
    list2 = [1, 1, 2, 4, ...]
    list3 = [4, 2, 2, 4, ...]
    list4 = [2, 4, 8, 1, ...]

    # Create a DataFrame
    data = {
        'Vocabulary 1': list1,
        'Vocabulary 2': list2,
        'Vocabulary 3': list3,
        'Vocabulary 4': list4
    }
    df = pd.DataFrame(data)

    # Create subplots for each vocabulary
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
    fig.suptitle('Token Length Distribution for Different Vocabularies')

    # Plot histograms for each vocabulary
    for i, (col, ax) in enumerate(zip(df.columns, axes.flat)):
        df[col].plot(kind='hist', ax=ax, bins=10, title=col)
        ax.set_xlabel('Token Length')
        ax.set_ylabel('Frequency')

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Show the plot
    plt.show()
