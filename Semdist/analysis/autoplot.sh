
# list of files name to plot: wer, semdist, random

plotlist="wer semdist random"

# print each element of the list
for plot in $plotlist
do
    cut -d" " -f1,2,16,17,19,20 ../results_lia_asr/$plot/1234/train_log.txt > data/$plot.txt
    python plot.py $plot
done

python plot.py $plotlist