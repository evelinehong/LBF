import json
from scipy import stats

file = "Math_23K"

with open(file + ".json", "r") as read_file:
    data = json.load(read_file)
    filtered = sorted(filtered, key=lambda item: len(item["equation"]))
    with open(file + "_sorted.json", "w") as write_file:
        json.dump(filtered, write_file, indent=4)