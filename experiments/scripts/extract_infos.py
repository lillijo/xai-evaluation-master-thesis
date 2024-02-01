import re
from os import listdir
from os.path import isfile, join
import json

tester = "best loss:  0.6931474804878235  last loss:  0.6931474804878235  best epoch:  model_0_0.11((((b0_11-s0_5-l0_009:{'train_accuracy': [100.0, 0.0, 49.72330856323242], 'all_wm_accuracy': [100.0, 0.0, 49.739585876464844], 'no_wm_accuracy': [100.0, 0.0, 49.666343688964844], 'bias': 0.11, 'strength': 0.5, 'learning_rate': 0.009}))))cuda:0"
mypath = "../outputs/jobs"
onlyfiles = sorted([f for f in listdir(mypath) if isfile(join(mypath, f))])
namereg = re.compile(r"(\(\(\(\()([b.i0-9_-]*)(:)")

accuracies = {}

for f in onlyfiles:
    data = open(join(mypath, f), "r")
    lines = data.readlines()
    accs = list(
        filter(
            lambda x: x.startswith("(((("),
            lines,
        )
    )
    for a in accs:
        name = namereg.search(a)
        if name is not None:
            rname = name.group(2)
            rest = a.replace(name.group(), "").replace("))))", "").replace("'", "\"")
            new_value = json.loads(rest)
            if  rname not in accuracies or accuracies[rname]["file"] < f:
                if rname in accuracies:
                    print(rname,accuracies[rname]["file"], f)
                accuracies[rname] = new_value
                accuracies[rname]["file"] = f

with open("../outputs/final_models.json", "w") as fd:
    json.dump(accuracies, fd, indent=2)
