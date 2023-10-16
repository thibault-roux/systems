import pandas as pd
import matplotlib.pyplot as plt


def read_file(filename):
    with open(filename, "r") as f:
        vocab = f.readlines()
        vocab = [line.split()[0] for line in vocab]
    return vocab

# check if tokens in vocabulary for a BPE of a certain size are included in the vocabulary of a BPE of a larger size
def check_tokens():
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
def plot_histograms():
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




def percent_morph():
    morphemes = ["aa", "adam", "ae", "aen", "ai", "aï", "ail", "aim", "ain", "am", "an", "aon", "aou", "au", "aw", "ay", "aye", "bb", "ca", "cc", "cca", "cce", "cch", "cci", "cco", "ccu", "ccueil", "ccy", "ce", "ch", "ci", "co", "cqu", "ct", "cu", "cueil", "cy", "dd", "ds", "ea", "ean", "eau", "ect", "ed", "ee", "ée", "ef", "ei", "eil", "eim", "ein", "em", "emmm", "en", "enn", "ent", "er", "es", "eu", "eû", "ew", "ez", "ff", "ga", "ge", "geu", "geü", "gg", "gge", "ggi", "gh", "gi", "gn", "go", "gt", "gu", "gua", "gue", "guë", "güe", "gui", "ign", "iil", "il", "ill", "illaire", "ille", "illier", "im", "imm", "imma", "imme", "immi", "immo", "immu", "in", "ing", "ll", "lle", "mm", "mn", "nn", "oa", "oe", "oi", "oil", "om", "on", "ou", "ph", "pp", "ps", "pt", "qu", "qua", "qui", "rh", "rr", "rrh", "sc", "sca", "sce", "sch", "sci", "sco", "scu", "scy", "ss", "th", "tia", "tie", "tiel", "tien", "tient", "tieuse", "tieux", "tion", "tt", "tz", "uil", "um", "un", "uy", "ym", "yn"]
    tokenizer_type = ["sentencepiece", "bpe", "unigram"]
    systems = ["1000", "750", "500", "250"]
    for tokenizer in tokenizer_type:
        for system in systems:
            if tokenizer == "sentencepiece":
                filename = "results_lia_asr/wav2vec2_ctc_fr_bpe" + system + "_7k_without_space/1234/save/" + system + "_bpe.vocab"
            elif tokenizer == "bpe":
                filename = "results_lia_asr/wav2vec2_ctc_fr_bpe" + system + "_7k/1234/save/" + system + "_bpe.vocab"
            elif tokenizer == "unigram":
                filename = "results_lia_asr/wav2vec2_ctc_fr_unigram" + system + "_7k/1234/save/" + system + "_unigram.vocab"
            vocab = read_file(filename)
            # compute percentage of morphemes in the vocabulary
            morphemes_in_vocab = 0
            for morpheme in morphemes:
                if morpheme in vocab:
                    morphemes_in_vocab += 1
            print("Percentage of morphemes in vocabulary for system " + system + ": " + str(morphemes_in_vocab/len(morphemes)*100))





if __name__ == "__main__":
    # previous_main() 1
    # check_tokens()

    # previous_main() 2
    # plot_histograms()

    percent_morph()