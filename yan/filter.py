print("This code will remove empty reference from the test.csv file and create a new one name new_test.csv")

with open("test.csv", "r", encoding="utf8") as file:
    txt = "ID,duration,wav,wrd\n"
    next(file)
    for ligne in file:
        line = ligne[:-1].split(",")
        if line[3] != '':
            txt += ligne
        else:
            print(ligne, end="")

with open("new_test.csv", "w", encoding="utf8") as file:
    file.write(txt)
