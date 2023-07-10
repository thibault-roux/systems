from matplotlib import pyplot as plt




if __name__ == "__main__":
    
    tokenizer = "bpe500"
    number = "7k"

    if tokenizer != "" and number != "":
        tokenizer = tokenizer + "_"

    filename = "results_lia_asr/wav2vec2_ctc_fr_" + tokenizer + number + "/1234/train_log.txt"

    with open(filename, 'r') as f:
        lines = f.readlines()
        train_losses = []
        valid_losses = []
        for line in lines:
            if "train loss" in line:
                train_losses.append(float(line.split("train loss: ")[1].split(" ")[0]))
                valid_losses.append(float(line.split("valid loss: ")[1].split(",")[0]))

    # plot loss curves

    plt.plot(train_losses, label="train loss")
    plt.plot(valid_losses, label="valid loss")
    plt.title("Loss curves on " + tokenizer + number)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # limit x and y axis
    plt.ylim(0, 2.5)
    # plt.xlim(0, 10)
    plt.show()
    plt.savefig("plots/" + tokenizer + number + ".png")
    plt.clf()
