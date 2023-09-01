import pickle


if __name__ == "__main__":
    systems = ["wav2vec2_ctc_fr_1k", "wav2vec2_ctc_fr_3k", "wav2vec2_ctc_fr_7k", "wav2vec2_ctc_fr_bpe1000_7k", "wav2vec2_ctc_fr_bpe100_7k", "wav2vec2_ctc_fr_bpe1500_7k", "wav2vec2_ctc_fr_bpe150_7k", "wav2vec2_ctc_fr_bpe250_7k", "wav2vec2_ctc_fr_bpe500_7k", "wav2vec2_ctc_fr_bpe50_7k", "wav2vec2_ctc_fr_bpe650_7k", "wav2vec2_ctc_fr_bpe750_7k", "wav2vec2_ctc_fr_bpe900_7k", "wav2vec2_ctc_fr_xlsr_53_french", "wav2vec2_ctc_fr_xlsr_53"] 
    for system in systems:
        p_ctc = pickle.load(open("p_ctc_" + system + ".pkl", "rb"))[0].cpu()
        tensor = p_ctc.numpy()
        pickle.dump(tensor, open("tensor_" + system + ".pkl", "wb"))