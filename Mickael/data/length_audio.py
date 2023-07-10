
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
        if len(text) < 500:
            audios.append(audio)
            texts_length.append(len(text))
        else:
            print(audio, text)
            input()

avg_audio = sum(audios) / len(audios)
avg_text = sum(texts_length) / len(texts_length)
print("The average length of audio is: ", avg_audio)
print("The average length of text is: ", avg_text)

print("The average speed of speech is: ", avg_text / avg_audio, "characters per second")
