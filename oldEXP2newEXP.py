import pickle
with open("dataset/EXP/raw/" + "GRAPHSAT" + ".pkl", "rb") as f:
    data_list = pickle.load(f)
new_data_list = [_.to_dict() for _ in data_list]
with open("dataset/EXP/raw/" + "newGRAPHSAT" + ".pkl", "wb") as f:
    pickle.dump(new_data_list, f)