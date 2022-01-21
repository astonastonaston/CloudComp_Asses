
with open("bst.txt", "r") as f:
    with open("res.txt", "w") as g:
        lines = f.readlines()
        # print(lines[1])
        for l in lines:

            # print(len(l))
            l = l.split(" ")
            # print(l)
            if ("Loss:" not in l): 
                continue
            else:
                ind = l.index("Perplexity:")
                # print(ind)
                # print(l[ind+1][:-1])
                g.write(str(float(l[ind+1][:-1])))
                g.write("\n")
        


