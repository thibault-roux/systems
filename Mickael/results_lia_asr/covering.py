
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
        paths.append("wav2vec2_ctc_fr_bpe" + vocabs[i].split("_")[0] + "_7k/1234/save/" + vocabs[i] + ".vocab")
    vocabs.append("250_morphemes")
    vocabs.append("750_morphemes")
    paths.append("wav2vec2_ctc_fr_bpe250_v3_7k/1234/save/250_bpe.vocab")
    paths.append("wav2vec2_ctc_fr_bpe750_v2_7k/1234/save/750_bpe.vocab")
    # modifier vocabs et ajouter le path complet en incluant les systèmes avec morphèmes
    txt = ""
    for v in vocabs:
        txt += v + ","
    txt = txt[:-1] + "\n"
    
    for i in range(len(vocabs)):
        vocab1 = vocabs[i]
        txt += vocab1 + ","
        for j in range(len(vocabs)):
            vocab2 = vocabs[j]
            path1 = paths[i]
            path2 = paths[j]
            if path1 != path2:
                inside, length = compute_covering(path1, path2)
                txt += str(inside/length*100) + ","
        txt = txt[:-1] + "\n"

    with open("coverage.txt", "w", encoding="utf8") as file:
        file.write(txt)