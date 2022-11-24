
import argparse
parser = argparse.ArgumentParser(description="Transform dataset that have my personal URL to the JeanZay")
parser.add_argument("filename", type=str, help="Path to the file to transform.")
args = parser.parse_args()

filename = args.filename
with open(filename, "r", encoding="utf8") as file:
    txt = next(file)
    for ligne in file:
        line = ligne.split(",")
        url = line[2]
        txt += line[0] + "," + line[1] + ",/gpfswork/rech/rbg/ujc17nz/corpus/" + url[20:] + "," + line[3]

with open("new_" + filename, "w", encoding="utf8") as file:
    file.write(txt)

