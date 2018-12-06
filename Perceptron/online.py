import numpy as np

def split(s, delim=[" ", '\n']):
    words = []
    word = []
    for c in s:
        if c not in delim:
            word.append(c)
        else:
            if word:
                words.append(''.join(word))
                word = []
    if word:
        print(word)
        words.append(''.join(word))
    return words

def loadfile(filename):
    file = open(filename, "r")
    first = True
    rows = list()
    for line in file:
        if(first) == True:
            dims = split(line)
            first = False
        else:
            vals = split(line, [' ' ,'\t', '\n'])
            #print(vals)
            rows.append(vals)

    return dims, rows



dims, rows = loadfile('Train.txt')

dims=np.array(dims)
dims = dims.astype(np.float)
rows=np.array(rows)
mat = rows.astype(np.float)
print(dims)
print(mat)



