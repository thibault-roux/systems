

word = input("Which word are you looking for? ")
# Measure length of audio and number of character in order to compute the speed of speech
with open("test.csv", "r", encoding="utf8") as file:
    audios = []
    texts_length = []
    next(file)
    for line in file:
        line = line.split(",")
        audio = float(line[1])
        for c in "() ":
            text = line[-1][:-1].replace(c, "")
        if word in text:
            print(line)
            print(audio, text)