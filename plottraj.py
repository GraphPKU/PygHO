import argparse
import numpy as np
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('infile', type=str)
args = parser.parse_args()
losss = []
score = []
with open(args.infile, "r") as f:
    for line in f.readlines():
        if "loss" in line:
            line = line.split()
            losss.append([float(line[-3]), float(line[-2]), float(line[-1])])
        elif "Validation" in line:
            line = line.replace("}", " ")
            tscore = []
            for ls in line.split():
                try:
                    tscore.append(float(ls))
                except:
                    pass
            score.append(tscore)
losss = np.array(losss)
score = np.array(score)
x = np.arange(losss.shape[0])
plt.figure()
plt.plot(x, losss[:, 0], label="loss")
plt.plot(x, losss[:, 1], label="policy")
plt.plot(x, losss[:, 2], label="negentropy")
plt.legend()
plt.yscale("symlog")
plt.savefig("loss.pdf")

plt.figure()
plt.plot(x, score[:, 0], label="valid")
plt.plot(x, score[:, 1], label="test")
plt.legend()
plt.savefig("score.pdf")