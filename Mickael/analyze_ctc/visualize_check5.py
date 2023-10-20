import numpy as np
import pickle
import matplotlib.pyplot as plt


def load_dict():
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


def compute_matrix(system, ind2tok, tok2ind, average=False, X=None, Y=None):
    tensor = pickle.load(open("pickle4/tensor_" + system + ".pkl", "rb"))[0]
    
    # print(tensor.shape) # (42, 900), there is 42 segments and 900 tokens in the vocabulary
    # I want the rank for each segment
    tensor_rank = np.zeros((tensor.shape[0], tensor.shape[1]))
    for i in range(tensor.shape[0]):
        tensor_rank[i] = np.argsort(tensor[i])[::-1] # reverse the array

    return [tensor_rank.mean(axis=0)]

    
    

def process(min_batch, max_batch, step_batch, mult, figx, figy):
    # Load dict
    ind2tok, tok2ind = load_dict()

    from matplotlib.colors import LinearSegmentedColormap
    colors = [(0, 'white'), ((1 - np.log(2))*1, 'darkblue'), (1, 'black')]
    custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)

    x_ticks = [i for i in range(min_batch, max_batch, step_batch)]
    systems = [str(i) for i in range(min_batch, max_batch, step_batch)] # (0, 36250, 50)]
    matrix = np.empty((900, 0))
    for system in systems:
        print(system)
        minimatrix = np.array([])
        for iteration in range(1, 21): # compute the average of the 20 iterations
            map2 = compute_matrix(system + "_" + str(iteration), ind2tok, tok2ind)
            try:
                minimatrix = np.concatenate((minimatrix, map2), axis=0)
            except ValueError:
                minimatrix = np.array(map2)
        
        minimatrix = minimatrix.mean(axis=0)
        matrix = np.concatenate((matrix, minimatrix.reshape(-1, 1)), axis=1)


    token_lengths = [len(ind2tok[x]) for x in range(matrix.shape[0])]
    sorted_indices = np.argsort(token_lengths)
    sorted_matrix = matrix[sorted_indices, :] # sort

    #sorted_matrix = sorted_matrix.T
    
    # Create the heatmap
    
    # plt.figure(figsize=(figx*mult, figy*mult))  # Adjust the figure size as needed
    # plt.imshow(sorted_matrix, cmap=custom_cmap, aspect='auto')
    # plt.colorbar()

    counter = dict()
    for i in range(len(token_lengths)):
        try:
            counter[token_lengths[i]] += 1
        except KeyError:
            counter[token_lengths[i]] = 1

    # print(sorted_matrix.shape) # (900, 3) car 900 tokens et 3 * 20 audio segments moyenné (on a prédit uniquement à 3 état neuronal)

    # Add horizontal grid lines to separate tokens
    average_per_length = dict()
    previous = 0
    for token_length in range(1, max(counter.keys())):
        count = counter[token_length]
        previous = previous + count
        plt.axhline(y=previous, color='red', linewidth=4)
        # print(previous, count, previous+count)
        # print(sorted_matrix[previous:previous+count].shape)
        subset = sorted_matrix[previous:previous+count]
        average_per_length[token_length] = subset.mean(axis=0)
        # if token_length == 1:
        #     print(average_per_length[token_length])
        #     print(average_per_length[token_length].shape)



    # # Set axis labels
    # plt.xlabel('Batch trained')
    # plt.ylabel('Tokens')
    # # Show the plot
    # plt.title('Heatmap of Probability Matrix')
    # plt.show()
    # plt.savefig("results/many_rank_heatmap_average.png")
    
    # plt.clf()

    # Extract data and convert it into a 2D NumPy array
    data = np.array(list(average_per_length.values()))
    print("data.shape:", data.shape)
    plt.imshow(data, cmap='viridis', aspect='auto', interpolation='none')
    plt.colorbar(label='Value')
    plt.xlabel('Time Step')
    plt.ylabel('Token Length')
    plt.title('Heatmap of Average per Length')
    plt.yticks(np.arange(len(average_per_length)), np.arange(1, len(average_per_length)+1))
    plt.xticks(np.arange(len(x_ticks)), x_ticks)
    plt.show()
    plt.savefig("results/many_rank_heatmap_average_per_length2.png")




if __name__ == "__main__":
    process(min_batch=0, max_batch=36250, step_batch=50*300, mult=10, figx=10, figy=14)

    # Every 50 batch, we infer the ASR system on 20 audio segments.
    # Instead of using the output of the ASR system, we compute the rank of the output.
    # We compute the average rank prediction by token length (for example, for tokens of length 1 at timestamp 1)
    # This heatmap shows the output (rank normalization) for each type of token. A token type is the length of the token so there is 10 types since a token can be long of max 10 characters.