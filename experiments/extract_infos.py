import re
from os import listdir
from os.path import isfile, join
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("accs_name")
    parser.add_argument("accs_name", help="accs_name string", type=str)
    args = parser.parse_args()

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
                rest = a.replace(name.group(), "").replace("))))", "").replace("'", '"')
                new_value = json.loads(rest)
                if (
                    rname not in accuracies
                    or new_value["train_accuracy"][2]
                    > accuracies[rname]["train_accuracy"][2]
                ):
                    if rname in accuracies:
                        print(
                            rname,
                            f,
                            accuracies[rname]["file"],
                            accuracies[rname]["train_accuracy"][2],
                            new_value["train_accuracy"][2],
                        )
                    accuracies[rname] = new_value
                    accuracies[rname]["file"] = f

    with open(f"../outputs/{args.accs_name}.json", "w") as fd:
        json.dump(accuracies, fd, indent=2)


