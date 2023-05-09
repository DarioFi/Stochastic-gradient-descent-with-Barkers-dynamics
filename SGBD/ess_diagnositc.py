import json

import matplotlib.pyplot as plt
import seaborn as sns

with open("ess_logs.json", "r") as file:
    data = json.load(file)

# todo: ask about difference bulk tail for ess

assert isinstance(data, list)

data.sort(key=lambda x: x["model"])
for x in data:
    # sns.histplot(x["bulks"])
    # sns.histplot(x["tails"])
    # plt.show()

    # del x["bulks"]
    # del x["tails"]

    if True:
        keys = ["model", "size", "bulk_min", "bulk_median", "tail_min"]
        print(x["model"], end=" & ")
        print(x["size"], end=" & ")
        print(f'{round(x["select_prob"] * 100, 1)}\%', end=" & ")
        print(f'{round(x["bulk_min"], 1)}, {round(x["bulk_median"], 1)}', end=" & ")
        print(f'{round(x["tail_min"], 1)}, {round(x["tail_median"], 1)}', end=" \\\\")
        print("")

# with open("ess_logs.json", "w") as file:
#     json.dump(data, file, indent=4)