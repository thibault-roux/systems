import numpy as np
import pickle
import matplotlib.pyplot as plt


def load_dict(system):
    ind2tok = dict()
    tok2ind = dict()
    ind = 0
    with open("../results_lia_asr/wav2vec2_ctc_fr_bpe900_CHECK_7k/1234/save/900_bpe.vocab", "r") as f:
        for line in f:
            tok = line.split()[0]
            tok = tok.replace("▁", " ")
            ind2tok[ind] = tok
            tok2ind[tok] = ind
            ind += 1
    return ind2tok, tok2ind


def get_lengths(system, batch_size): # vocab = {token, char, all}, normtype = {exp, norm}
    tensor = pickle.load(open("pickle3/tensor_" + system + ".pkl", "rb"))[0]

    # Load dict
    ind2tok, tok2ind = load_dict(system)
    
    length = []
    for i in range(tensor.shape[0]):
        # select the indice of the highest values of the numpy array that correspond to a token of length 1
        indices = np.argsort(tensor[i])
        indices = indices[::-1] # reverse the array
        minibatch = []
        # if ind2tok[indices[0]] != "<unk>":
        #     continue
        for indice in indices:
            # if len(ind2tok[indice]) == 1:
            if indice in ind2tok.keys() and ind2tok[indice] != "<unk>": # and ind2tok[indice] in useful_toks:
                # print(i, ind2tok[indice], end="\n")
                minibatch.append(len(ind2tok[indice]))
                if len(minibatch) > batch_size:
                    length.append(np.mean(minibatch))
                    break

    return length

if __name__ == "__main__":

    for batch_size in [1,100]:

        total_length = []

        systems = [str(i) for i in range(0, 15050, 1)] # (0, 15050, 50)]
        for system in systems:
            print(system)
            total_length.extend(get_lengths(system, batch_size=batch_size)) # vocab = {token, char, all}, normtype = {exp, norm}

        # scatter plot the length
        x = range(len(total_length))
        plt.scatter(x, total_length, s=0.1)
        
        # Calculate the smoothed line by averaging nearby points (e.g., using a moving average)
        window_size = 2000
        smoothed_total_length = np.convolve(total_length, np.ones(window_size)/window_size, mode='valid')
        # Adjust x values for the smoothed line to match the valid region after convolution
        smoothed_x = x[(window_size-1)//2:-(window_size-1)//2]
        # Plot the smoothed line
        plt.plot(smoothed_x, smoothed_total_length, label='Smoothed Line', color='red')


        plt.savefig("length_FULL_" + str(batch_size) + ".png")
        # clear plt
        plt.clf()