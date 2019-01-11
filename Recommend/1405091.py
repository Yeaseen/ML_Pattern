import numpy as np
import time
from sklearn.model_selection import train_test_split


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
        pos=np.where(rM!=99)
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

   

train=loadfile('data.txt')
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

users=len(train)
print(users)
items=len(train[0])
print(items)    

latent_factors=[5,10,20,40]
lambdaSet=[0.01,0.1,1,10]
Ulist=[[]]
Vlist=[[]]
unik=0
lam=0
pre_val_err=np.finfo(np.float).max

start = time.time()
f=open("uvSet.txt","w")
for k in latent_factors:
    for lambdau in lambdaSet:
        U=np.random.uniform(-100,100,(users,k))
        V=np.zeros((k,items))
        prev_error=0
        while(True):
            for i in range(items):
                #print(i)
                ux=trainSet[::,i]
                pstns=np.where(ux !=99)
                pstns=pstns[0]
                invpart=np.zeros((k,k))
                otpart=np.zeros((k,1))
                
                for j in pstns:
                    mx=U[j]
                    mx=mx.reshape(1,k)
                    my=np.transpose(mx)
                    ans=np.matmul(my,mx)
                    invpart=invpart+ans
                    otpart=otpart+ux[j]*my
                
                fans=invpart+lambdau*np.identity(k)    
                fans=np.matmul(np.linalg.inv(fans),otpart)
                V[::,i]=np.transpose(fans)
                #break
            for n in range(users):
                ux=trainSet[n]
                #print(vx)
                pstns=np.where(ux !=99)
                pstns=pstns[0]
                invpart=np.zeros((k,k))
                otpart=np.zeros((k,1))
                
                for j in pstns:
                    mx=V[::,j]
                    #print(mx)
                    mx=mx.reshape(1,k)
                    my=np.transpose(mx)
                    ans=np.matmul(my,mx)
                    invpart=invpart+ans
                    otpart=otpart+ux[j]*my
                fans=invpart+lambdau*np.identity(k)    
                fans=np.matmul(np.linalg.inv(fans),otpart)
                U[n]=np.transpose(fans)
                #break
            
            curr_err=RMSE(U,V,trainSet)
            #print(curr_err)
            if(np.abs(prev_error-curr_err)<0.01):
                break
            prev_error=curr_err
        
        valRMSE=RMSE(U,V,valSet)
        if(valRMSE<pre_val_err):
            pre_val_err=valRMSE
            Ulist=U
            Vlist=V
            unik=k
            lam=lambdau
        f.write('For K= '+str(k)+'     lambda= '+str(lambdau)+'      '+str(valRMSE)+'\n')
f.close()
print("Training finished, time needed: ", time.time() - start)



rmseTest=RMSE(Ulist,Vlist,testSet)


print('For K= '+str(unik)+'     lambda= '+str(lam)+'      '+str(rmseTest))
finalData=np.matmul(Ulist,Vlist)






















