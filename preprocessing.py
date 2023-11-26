import regex as re
from collections import defaultdict
import json

def create_labels(file = "./FSD50K.ground_truth/dev.csv"):
    """
    Create a dictionary of sound labels as keys where each label is a sound class and the values 
    are a list of tuples containing the file name and split (train, val). 
    Sound class assigned is the first class listed in the labels column of the csv file.
    Other classes present in labels column are stored in another dictionary of paired classes.
    """
    res = defaultdict(list)
    paired_classes = defaultdict(list)
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            try:
                regex = r'".+?"|[\w-]+' # regex to ignore , within quotes
                fname, labels, mids, split = re.findall(regex, line) 
                labels = labels[1:-1]
            except:
                fname, labels, mids, split = line.split(",")
            classes = labels.split(",")
            primary_class = classes[0] # primary class is the first class listed
            if len(classes) > 1:
                secondary_classes = classes[1:] if len(classes) > 1 else []
                res[primary_class].append((fname))
                paired_classes[primary_class].append(secondary_classes)
            else:
                res[primary_class].append((fname))

    return res, paired_classes  

labels, paired_classes = create_labels()
json_labels = json.dumps(labels)
with open("labels.json", "w") as f:
    f.write(json_labels)

with open("files_per_class.txt", "w") as f:
    for l in labels:
        f.write(l + "," + str(len(labels[l])) + "\n")
        