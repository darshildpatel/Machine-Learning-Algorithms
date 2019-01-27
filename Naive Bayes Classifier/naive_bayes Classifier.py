import pandas as pd
import math

dataset = pd.read_csv('spambasetrain.csv',header = None)
testset = pd.read_csv('spambasetest.csv',header = None)

def class_prob(dataset, label):
    #for i in dataset[column].unique():
        a = dataset[dataset[9] == label]
        prob = a[9].count()/dataset[9].count()
        #print("Probability of",label,prob)
        return prob  
    
def get_variance(dataset, column, label):
    a = dataset[dataset[9] == label][column]
    mean = get_mean(dataset, column, label)
    variance = 0
    for row in a:
        variance = variance + pow(row - mean,2)
    #print(variance/a.count())
    return math.sqrt(variance/(a.count()-1))   

def get_mean(dataset, column, label):
    a = dataset[dataset[9] == label][column]
    #print(a.mean())
    return a.mean()

def likelihood(mean, variance, x):
    a = math.exp((-pow(x-mean,2))/(2*pow(variance,2)))
    b = math.sqrt(44*pow(variance,2)/7)
    return a/b

def pdf(dataset, x, column, label):
    req_column = dataset[dataset[9] == label][column]
    prior = class_prob(dataset, label)
    mean = get_mean(dataset, column, label)
    variance = get_variance(dataset, column, label)
    num = likelihood(mean, variance, x)     
    
    return num

def apply_NB(a, label):
    num = 0
    for i in range(0,8):
        num = num + math.log(pdf(dataset, a[i], i, label))
        
    return num + math.log(class_prob(dataset, label))

def predict():
    a = []
    for index, row in testset.iterrows():
#         print(apply_NB(row, 0))
#         print(apply_NB(row, 1))
        if apply_NB(row, 0) > apply_NB(row, 1):
            a.append(0)
        else:
            a.append(1)
    
    return a  

def accuracy_score(true, predictions, parameter):
    a,b =0,0
    for r,y in zip(true,predictions):
        if r == y:
            a = a+1
        else:    
            b = b+1
            
    if parameter == 'correct':
        return a
    if parameter == 'incorrect':
        return b
    else:
        return (a/len(true))*100 
    
def output():#this is the main function
    predictions = predict()
    print("P(0) =",class_prob(dataset, 0))
    print("P(1) =",class_prob(dataset, 1))
    for j in [0,1]:
        for i in range(0,9):
            print("Mean, Variance for class {} and column {}= {},{}".format(j, i+1, get_mean(dataset, i, j),get_variance(dataset, i, j)))
            
    print("The output of all the testset examples is as follows",predictions)  
    print("Total number of test examples classified correctly = ",accuracy_score(testset[9],predictions,'correct'))
    print("Total number of test examples classified incorrectly = ",accuracy_score(testset[9],predictions,'incorrect'))
    print("The percentage error on the test examples = {}%".format(100 - accuracy_score(testset[9],predictions,'score')))    
      
def main():
	output()
	
if __name__ == "__main__":
	main()
	


