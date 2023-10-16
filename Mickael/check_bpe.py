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
    dfs = []
    systems = ["1000", "750", "500", "250"]
    for system in systems:
        filename = "results_lia_asr/wav2vec2_ctc_fr_bpe" + system + "_7k_without_space/1234/save/" + system + "_bpe.vocab"
        vocab = read_file(filename)
        token_lengths = [len(token) for token in vocab]
        df = pd.DataFrame({'System': f'System {system}', 'Token Length': token_lengths})
        dfs.append(df)

    # Combine DataFrames into a single DataFrame
    combined_df = pd.concat(dfs)

    colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']

    # Create a histogram plot
    reduce = 0.8
    plt.figure(figsize=(8*reduce, 6*reduce))
    plt.hist([df['Token Length'] for df in dfs], bins=range(1, 11), rwidth=0.8, label=[df['System'].iloc[0] for df in dfs], align='left', color=colors)
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.xticks(range(1, 11))
    plt.legend()

    # Show the plot
    plt.title('Token Length Distribution for Different Vocabularies')

    # Show the plot
    plt.show()
    plt.savefig("plots/hist_sentencepiece_token_length.svg")
