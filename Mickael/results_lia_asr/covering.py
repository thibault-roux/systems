
def read_vocab(file):
    vocab = []
    with open(file, "r") as f:
        for line in f:
            vocab.append(line.split()[0])
    return vocab

def compute_covering(file1, file2):
    v1 = read_vocab(file1)
    v2 = read_vocab(file2)

    inside = 0
    for w in v1:
        if w in v2:
            inside += 1
    return inside, len(v1)

if __name__ == "__main__":
    vocabs = ["100_char", "70_bpe", "100_bpe", "150_bpe", "250_bpe", "500_bpe", "650_bpe", "750_bpe", "900_bpe", "1000_bpe", "1500_bpe"]
    paths = ["wav2vec2_ctc_fr_7k/1234/save/100_char.vocab", "wav2vec2_ctc_fr_bpe50_7k/1234/save/70_bpe.vocab"]
    for i in range(2, len(vocabs)):
        paths.append("wav2vec2_ctc_fr_bpe" + vocab[i].split("_")[0] + "_7k/1234/save/" + vocabs[i] + ".vocab")
    vocabs.append("250_morphemes")
    vocabs.append("750_morphemes")
    paths.append("wav2vec2_ctc_fr_bpe250_v3_7k/1234/save/250_bpe.vocab")
    paths.append("wav2vec2_ctc_fr_bpe750_v2_7k/1234/save/750_bpe.vocab")
    # modifier vocabs et ajouter le path complet en incluant les systèmes avec morphèmes
    for v1 in vocabs:
        for v2 in vocabs:
            v1 = path + v1 + ".vocab"
            v2 = path + v2 + ".vocab"
            if v1 != v2:
                inside, length = compute_covering("vocabularies/" + v1 + ".txt", "vocabularies/" + v2 + ".txt")
                print(v1, v2, inside/length*100)