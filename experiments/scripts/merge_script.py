import json
import argparse

def merge_files(file_a, file_b):
    with open(f"../outputs/{file_a}.json", "r") as f:
        accuracies_a = json.load(f)
    
    with open(f"../outputs/{file_b}.json", "r") as f:
        accuracies_b = json.load(f)
    
    for name, item in (accuracies_b.items()):
        if name not in accuracies_a:
            accuracies_a[name] = item

    with open(f"../outputs/{file_a}.json", "w") as f:
        json.dump(accuracies_a, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("script_parallel")
    parser.add_argument("file_a", help="bias float", type=str)
    parser.add_argument("file_b", help="bias float", type=str)
    args = parser.parse_args()
    merge_files(args.file_a, args.file_b)