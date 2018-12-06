
import csv
import math



fund, sal, pur =[], [], []

with open('Data.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    for line in csv_reader:
        fund.append(line[0])
        sal.append(line[1])
        pur.append(line[2])
    csv_file.close()

print(fund)
print(sal)
print(pur)

length = len(sal)
pos=0
neg=0

for val in range(0, length):
    if(pur[val]=="1"): pos+=1
    else: neg+=1

posProb=pos/length
negProb=neg/length

def mean(sal,str):
    sum = 0
    rc = length
    count = 0
    for val in range(0, rc):
        if (pur[val] == str):
            sum += float(sal[val])
            count+=1
    res = sum / count
    return res

def stdev(sal,str,mean):
        sum = 0
        rc = length
        count = 0
        for val in range(0, rc):
            if (pur[val] == str):
                sum += pow(abs(float(sal[val]) - mean), 2)
                count+=1
        res = sum / count
        std = math.sqrt(res)
        return std

def normD(X,mean,stdV):
    res1=pow(abs(X-mean),2)
    res2=(2*pow(stdV,2))
    res = res1 / res2
    ans=math.exp(-res)/(math.sqrt(2*3.1416)*stdV)
    return ans


def desProb(array, query, output):
    querycount=0
    totalCount=0
    for val in range(0,length):
        if(pur[val]==output):
            totalCount+=1
            if(array[val]==query):
                querycount+=1
    #print(querycount)
    #print(totalCount)
    res=querycount / totalCount
    return res


#m=mean(sal,"0")

#sd=stdev(sal,"0",m)


#data = 120000

#NrmDist=normD(int(data),m,sd)

#descreteProb=desProb(fund,"1","0")

#print("Mean is : "+str(m))
#print("Standard deviance is : " + str(sd))
#print("Normal Distribution is : "+ str(NrmDist))
#print("Sample neg prob: " + str(negProb))

#print("fund=Yes, purchase=No: "+ str(descreteProb))

funTest, salTest, testOut, OutPut = [], [], [], []

with open('test.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)


    for line in csv_reader:
        funTest.append(line[0])
        salTest.append(line[1])
        testOut.append(line[2])
    csv_file.close()

lengthTest=len(funTest)



for val in range(0, lengthTest):
    refund=funTest[val]
    #print(refund)
    salary=salTest[val]
    #print(salary)
    fetOneProbNeg=desProb(fund,refund,'0')
    mean1=mean(sal,'0')
    #print(mean)
    stdeviance=stdev(sal,'0',mean1)
    #print(stdeviance)
    fetTwoProbNeg=normD(float(salary),mean1,stdeviance)
    #print(fetOneProb)
    #print(fetTwoProb)
    negativeCheck= negProb*fetOneProbNeg*fetTwoProbNeg
    print(negativeCheck)

    fetOneProbPos=desProb(fund,refund,'1')
    mean2=mean(sal,'1')
    stddeviance2=stdev(sal,'1',mean2)
    fetTwoProbPos=normD(float(salary), mean2,stddeviance2)
    positiveCheck= posProb*fetOneProbPos*fetTwoProbPos
    print(positiveCheck)

    if(positiveCheck>negativeCheck):
        OutPut.append('1')
    elif(negativeCheck>positiveCheck):
        OutPut.append('0')
    else:
        OutPut.append('1')






with open('output.csv', 'w') as csvoutput:
    writer = csv.writer(csvoutput)
    all =['Refund','Income','testOutput','FuncOutput']
    writer.writerow(all)
    for row in range(0, lengthTest):
        all = []
        all.append(funTest[row])
        all.append(salTest[row])
        all.append(testOut[row])
        all.append(OutPut[row])
        writer.writerow(all)

    csvoutput.close()
