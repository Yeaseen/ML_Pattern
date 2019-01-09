import numpy as np
from sklearn.model_selection import train_test_split
from scipy import linalg
def split(s, delim):
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
        words.append(''.join(word))
    return words

def loadfile(filename):
    file = open(filename, "r")
    rows = list()
    for line in file:
        vals = split(line, [' ',',' ,'\t', '\n'])
        rows.append(vals)
    return rows

def RMSE(U,V,TestM):
    outM=np.matmul(U,V)
    count=0
    result=0
    for rM,rO in zip(TestM,outM):
        pos=np.where(rM!=2)
        pos=pos[0]
        count=count+len(pos)
        for j in pos:
            result=result+np.power(rM[j]-rO[j],2)
    result=result/count
    result=np.sqrt(result)
    return result


#u=[[1,2],[3,4]]
#v=[[1,2],[3,4]]
#w=[[1,2],[3,4]]

#print(RMSE(np.array(u),np.array(v),np.array(w)))

   

train=loadfile('dd.txt')
train=np.array(train)
train= train.astype(np.float)
train = np.delete(train, 0, axis=1)

trainSet=train.copy()
valSet=train.copy()
testSet=train.copy()


for k in range(len(train)):
    #print(k)
    data=train[k]
    poss=np.where(data != 99)
    poss=poss[0]
    x_train,x_val_test = train_test_split(poss,test_size=0.4,random_state=42)
    for i in x_val_test:
        trainSet[k][i]=99
    for i in x_train:
        valSet[k][i]=99
        testSet[k][i]=99
    x_val,x_test = train_test_split(x_val_test,test_size=0.5,random_state=42) 
    for i in x_test:
        valSet[k][i]=99
    for i in x_val:
        testSet[k][i]=99

#k=latent factor
k=5
users=len(train)
print(users)
items=len(train[0])
print(items)    

U=np.random.randint(-10,10,(users,k))
V=np.zeros((k,items))

i=0
ux=train[::,i]
pstns=np.where(ux !=99)
pstns=pstns[0]

#np.identity(3)
invpart=np.zeros((k,k))
otpart=np.zeros((k,1))
lambdau=0.1

for i in pstns:
    mx=U[i]
    mx=mx.reshape(1,5)
    my=np.transpose(mx)
    #print(ux[i]*my)
    ans=np.matmul(my,mx)
    invpart=invpart+ans
    otpart=otpart+ux[i]*my
fans=invpart+lambdau*np.identity(k)    
fans=np.matmul(np.linalg.inv(fans),otpart)    
print(fans)    

#ux=np.array([0,1,2,3,4])
#V[::,3] = np.transpose(ux)






















