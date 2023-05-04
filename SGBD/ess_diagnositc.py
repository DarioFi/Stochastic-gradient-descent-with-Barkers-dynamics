import json

import matplotlib.pyplot as plt
import seaborn as sns

with open("ess_logs.json", "r") as file:
    data = json.load(file)

# todo: ask about difference bulk tail for ess

for x in data:
    sns.histplot(x["bulks"])
    sns.histplot(x["tails"])
    plt.show()

    del x["bulks"]
    del x["tails"]
    print(x)
