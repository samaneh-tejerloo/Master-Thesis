path = "datasets/golden standard/ada_ppi.txt"

with open(path, "r") as f:
    lines = f.readlines()
# %%
complexes = [line.split()[:-1] for line in lines]
complexes
# %%
for i in range(len(complexes)):
    for j in range(len(complexes)):
        if i != j:
            a = set(complexes[i])
            b = set(complexes[j])
            intersect = a.intersection(b)
            if len(intersect) == min(len(a), len(b)):
                print(i, j)
                break

complexes[749]
complexes[29]
