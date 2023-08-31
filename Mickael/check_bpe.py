

def read_file(filename):
    with open(filename, "r") as f:
        vocab = f.readlines()
        vocab = [line.split()[0] for line in vocab]
    return vocab

# check if tokens in vocabulary for a BPE of a certain size are included in the vocabulary of a BPE of a larger size
if __name__ == "__main__":
    systems = ["50", "100", "150", "250", "500", "650", "750", "900", "1000", "1500"]
    vocab2 = []
    for system in systems:
        print("System processed: " + system)
        if system == "50":
            system2 = "70"
        else:
            system2 = system
        filename = "results_lia_asr/wav2vec2_ctc_fr_bpe" + system + "_7k/1234/save/" + system2 + "_bpe.vocab"
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