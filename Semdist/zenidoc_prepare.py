"""
Data preparation.

Download: https://voice.mozilla.org/en/datasets

Author
------
Titouan Parcollet
"""

import os
import csv
import re
import logging
import torchaudio
import unicodedata
from tqdm import tqdm
from tqdm.contrib import tzip

from pqdm.processes import pqdm

logger = logging.getLogger(__name__)


def prepare_zenidoc(
    data_folder,
    save_folder,
    train_tsv_file=None,
    dev_tsv_file=None,
    test_tsv_file=None,
    accented_letters=True,
    language="fr",
    skip_prep=False,
):
    """
    Prepares the csv files for the Mozilla Common Voice dataset.
    Download: https://voice.mozilla.org/en/datasets

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Common Voice dataset is stored.
        This path should include the lang: /datasets/CommonVoice/en/
    save_folder : str
        The directory where to store the csv files.
    train_tsv_file : str, optional
        Path to the Train Common Voice .tsv file (cs)
    dev_tsv_file : str, optional
        Path to the Dev Common Voice .tsv file (cs)
    test_tsv_file : str, optional
        Path to the Test Common Voice .tsv file (cs)
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    language: str
        Specify the language for text normalization.
    skip_prep: bool
        If True, skip data preparation.

    Example
    -------
    >>> from recipes.CommonVoice.px_corpus_prepare import prepare_px_corpus
    >>> data_folder = '/datasets/CommonVoice/en'
    >>> save_folder = 'exp/CommonVoice_exp'
    >>> train_tsv_file = '/datasets/CommonVoice/en/train.tsv'
    >>> dev_tsv_file = '/datasets/CommonVoice/en/dev.tsv'
    >>> test_tsv_file = '/datasets/CommonVoice/en/test.tsv'
    >>> accented_letters = False
    >>> duration_threshold = 10
    >>> prepare_px_corpus( \
                 data_folder, \
                 save_folder, \
                 train_tsv_file, \
                 dev_tsv_file, \
                 test_tsv_file, \
                 accented_letters, \
                 language="en" \
                 )
    """

    if skip_prep:
        return

    # If not specified point toward standard location w.r.t CommonVoice tree
    if train_tsv_file is None:
        train_tsv_file = data_folder + "/train.tsv"
    else:
        train_tsv_file = train_tsv_file

    if dev_tsv_file is None:
        dev_tsv_file = data_folder + "/dev.tsv"
    else:
        dev_tsv_file = dev_tsv_file

    if test_tsv_file is None:
        test_tsv_file = data_folder + "/test.tsv"
    else:
        test_tsv_file = test_tsv_file

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    # Setting ouput files
    save_csv_train = save_folder + "/train.csv"
    save_csv_dev = save_folder + "/dev.csv"
    save_csv_test = save_folder + "/test.csv"

    # # If csv already exists, we skip the data preparation
    # if skip_if_exist(save_csv_train, save_csv_dev, save_csv_test):

    #     msg = "%s already exists, skipping data preparation!" % (save_csv_train)
    #     logger.info(msg)

    #     msg = "%s already exists, skipping data preparation!" % (save_csv_dev)
    #     logger.info(msg)

    #     msg = "%s already exists, skipping data preparation!" % (save_csv_test)
    #     logger.info(msg)

    #     return

    # # Additional checks to make sure the data folder contains Common Voice
    # check_commonvoice_folders(data_folder)

    tokenizer_file = open(save_folder + "/train.txt","w")

    # Creating csv files for {train, dev, test} data
    file_pairs = zip(
        [train_tsv_file, dev_tsv_file, test_tsv_file],
        [save_csv_train, save_csv_dev, save_csv_test],
    )

    for tsv_file, save_csv in file_pairs:

        create_csv(
            orig_tsv_file=tsv_file,
            csv_file=save_csv,
            data_folder=data_folder,
            tokenizer_file=tokenizer_file,
            accented_letters=accented_letters,
            language=language,
        )

    tokenizer_file.close()

def skip_if_exist(save_csv_train, save_csv_dev, save_csv_test):

    # Checking folders and save options
    skip = False

    if (
        os.path.isfile(save_csv_train)
        and os.path.isfile(save_csv_dev)
        and os.path.isfile(save_csv_test)
    ):
        skip = True

    return skip

def process_line(line: str):

    client_id, path, sentence, up_votes, down_votes, age, gender, accents, locale, segment = line.split("\t")

    wav_path = path
    file_name = wav_path.split(".")[-2].split("/")[-1]
    spk_id = client_id
    snt_id = str(abs(hash(wav_path)))

    if os.path.isfile(wav_path):
        info = torchaudio.info(wav_path)
    else:
        msg = "\tError loading: %s" % (str(len(file_name)))
        logger.info(msg)
        return None

    duration = info.num_frames / info.sample_rate

    if duration <= 0.0:
        return None

    # total_duration += duration

    words = sentence

    # Unicode Normalization
    words = unicode_normalisation(words)

    words = words.lower()

    words = re.sub("\s+", " ", words)

    #words = words.replace("'", " ' ")
    #words = words.replace("’", " ’ ")

    words = words.lstrip().rstrip()

    # Getting chars
    chars = words.replace(" ", "_")
    chars = " ".join([char for char in chars][:])

    # Remove too short sentences (or empty):
    if len(words.split(" ")) < 3:
        return None

    # Composition of the csv_line
    csv_line = [snt_id, str(duration), wav_path, spk_id, str(words)]

    return csv_line

def create_csv(
    orig_tsv_file,
    csv_file,
    data_folder,
    tokenizer_file,
    accented_letters=True,
    language="fr",
):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    orig_tsv_file : str
        Path to the Common Voice tsv file (standard file).
    data_folder : str
        Path of the CommonVoice dataset.
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.

    Returns
    -------
    None
    """

    # Check if the given files exists
    if not os.path.isfile(orig_tsv_file):
        msg = "\t%s doesn't exist, verify your dataset!" % (orig_tsv_file)
        logger.info(msg)
        raise FileNotFoundError(msg)

    # We load and skip the header
    loaded_csv = open(orig_tsv_file, "r").readlines()[1:]
    nb_samples = str(len(loaded_csv))

    # tokenizer_file.write("\n".join(loaded_csv))

    msg = "Preparing CSV files for %s samples ..." % (str(nb_samples))
    logger.info(msg)

    # Adding some Prints
    msg = "Creating csv lists in %s ..." % (csv_file)
    logger.info(msg)

    csv_lines = [["ID", "duration", "wav", "spk_id", "wrd"]]

    if torchaudio.get_audio_backend() != "sox_io":
        logger.warning("This recipe needs the sox-io backend of torchaudio")
        logger.warning("The torchaudio backend is changed to sox_io")
        # torchaudio.set_audio_backend("soundfile")
        # torchaudio.set_audio_backend("sox_io")

    # Start processing lines
    total_duration = 0.0

    # for line in loaded_csv:
    # for line in loaded_csv:
    #     csv_line = process_line(line)

    # for l in tqdm(loaded_csv):
    #     process_line(l)
    
    # exit(0)
    
    print("@"*50)

    results = pqdm(loaded_csv, process_line, n_jobs=10)


    for r in results:
        if r != None: 
            total_duration += float(r[1])

            # Outputs
            csv_lines.append(r)
            tokenizer_file.write(r[-1] + "\n")


    # Writing the csv lines
    with open(csv_file, mode="w", encoding="utf-8") as csv_f:

        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final prints
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)
    msg = "Number of samples: %s " % (str(len(loaded_csv)))
    logger.info(msg)
    msg = "Total duration: %s Hours" % (str(round(total_duration / 3600, 2)))
    logger.info(msg)

# def check_commonvoice_folders(data_folder):
#     """
#     Check if the data folder actually contains the Common Voice dataset.

#     If not, raises an error.

#     Returns
#     -------
#     None

#     Raises
#     ------
#     FileNotFoundError
#         If data folder doesn't contain Common Voice dataset.
#     """

#     files_str = "/clips"

#     # Checking clips
#     if not os.path.exists(data_folder + files_str):

#         err_msg = (
#             "the folder %s does not exist (it is expected in "
#             "the Common Voice dataset)" % (data_folder + files_str)
#         )
#         raise FileNotFoundError(err_msg)


def unicode_normalisation(text):

    try:
        text = unicode(text, "utf-8")
    except NameError:  # unicode is a default on python 3
        pass
    return str(text)


def strip_accents(text):

    text = (
        unicodedata.normalize("NFD", text)
        .encode("ascii", "ignore")
        .decode("utf-8")
    )

    return str(text)
