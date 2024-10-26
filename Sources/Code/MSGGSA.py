from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import random
from itertools import combinations as cb
import math
from copy import deepcopy as dc
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import sys
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score



class Evaluate:
    def __init__(self, tr_x, tr_y):
        self.tr_x = tr_x
        self.tr_y = tr_y
        self.K = 5

    def evaluate(self, gen):
        # Applying feature selection
        #print(gen)
        gen=np.array(gen)
        selected_features = np.where(gen == 1)[0]
        al_data = self.tr_x[:, selected_features]
        #print(selected_features)

        # Initialize KFold and store a list of accuracy rates
        kf = KFold(n_splits=self.K, shuffle=True)
        scores = []


        # Cross validation
        for tr_ix, te_ix in kf.split(al_data, self.tr_y):
            # Note that we used al_data for training and testing
            knn_classifier = KNeighborsClassifier(n_neighbors=3, weights='distance')
            knn_classifier.fit(al_data[tr_ix], self.tr_y[tr_ix])
            score = knn_classifier.score(al_data[te_ix], self.tr_y[te_ix])
            scores.append(score)


            # The average accuracy and standard deviation are calculated
        avg_score = np.mean(scores)
        std_score = np.std(scores)


        # Output results
        print(f"average accuracy: {avg_score:.4f}, 标准差: {std_score:.4f}")

        # You can optionally output the fitness function values, but here only the average accuracy and standard deviation are returned
        selectfea_count = np.count_nonzero(gen == 1)
        f = 0.9 * avg_score + 0.1 * (1 - selectfea_count / len(gen))

        return f

    def check_dimentions(self, dim):  # check number of all feature
        if dim == None:
            return len(self.tr_x[0])
        else:
            return dim
'''class Evaluate:
    def __init__(self, tr_x, tr_y):
        self.tr_x = tr_x
        self.tr_y = tr_y
        self.K = 5
    def evaluate(self, gen):
        mask = np.array(gen) > 0
        al_data = self.tr_x[:, np.nonzero(mask)[0]]
        kf = KFold(n_splits=self.K, shuffle=True)
        s = 0
        scores = []
        for tr_ix, te_ix in kf.split(self.tr_x):
            knn_classifier = KNeighborsClassifier(n_neighbors=3,weights='distance')  # 使用KNN分类器，可以设置n_neighbors参数
            knn_classifier.fit(al_data[tr_ix], self.tr_y[tr_ix])
            s += knn_classifier.score(al_data[te_ix], self.tr_y[te_ix])
            score = knn_classifier.score(al_data[te_ix], self.tr_y[te_ix])
            scores.append(score)
        #print("fittness",s/self.K)
        selectfea_count = np.count_nonzero(gen == 1)
        f=0.9*s / self.K+0.1*(1-selectfea_count/len(gen))
        std_accuracy = np.std(scores)* 10**2
        
        #return std_accuracy
        return s / self.K
        #↑evaluate with fittness function

    def check_dimentions(self, dim):#check number of all feature
        if dim==None:
            return len(self.tr_x[0])
        else:
            return dim'''

'''#SVM classifier
class Evaluate:
    def __init__(self, tr_x, tr_y):
        self.tr_x = tr_x
        self.tr_y = tr_y
        self.K = 5

    def evaluate(self, gen):
        mask = np.array(gen) > 0
        al_data = self.tr_x[:, np.nonzero(mask)[0]]
        kf = KFold(n_splits=self.K, shuffle=True)
        s = 0
        scores = []
        for tr_ix, te_ix in kf.split(self.tr_x):
            svm_classifier = SVC()  # Use the SVM classifier
            svm_classifier.fit(al_data[tr_ix], self.tr_y[tr_ix])
            s += svm_classifier.score(al_data[te_ix], self.tr_y[te_ix])
            score = svm_classifier.score(al_data[te_ix], self.tr_y[te_ix])
            scores.append(score)

        selectfea_count = np.count_nonzero(gen == 1)
        f = 0.9 * s / self.K + 0.1 * (1 - selectfea_count) / len(gen)
        std_accuracy = np.std(scores) * 10 ** 2

        #return std_accuracy
        return s / self.K
    def check_dimentions(self, dim):#check number of all feature
        if dim==None:
            return len(self.tr_x[0])
        else:
            return dim'''

#DTclassifier
'''class Evaluate:
    def __init__(self, tr_x, tr_y):
        self.tr_x = tr_x
        self.tr_y = tr_y
        self.K = 5

    def evaluate(self, gen):
        mask = np.array(gen) > 0
        al_data = self.tr_x[:, np.nonzero(mask)[0]]
        kf = KFold(n_splits=self.K, shuffle=True)
        scores = []
        s = 0
        precision = 0
        recall = 0
        for tr_ix, te_ix in kf.split(self.tr_x):
            dt_classifier = DecisionTreeClassifier()  # Use a decision tree classifier
            dt_classifier.fit(al_data[tr_ix], self.tr_y[tr_ix])
            s += dt_classifier.score(al_data[te_ix], self.tr_y[te_ix])
            y_pred = dt_classifier.predict(al_data[te_ix])
            precision += precision_score(self.tr_y[te_ix], y_pred,average='micro')
            recall += recall_score(self.tr_y[te_ix], y_pred,average='micro')
            scores.append(dt_classifier.score(al_data[te_ix], self.tr_y[te_ix]))
            
 
        precision /= self.K
        recall /= self.K
        
        selectfea_count = np.count_nonzero(gen == 1)
        f = 0.9 * s / self.K + 0.1 * (1 - selectfea_count) / len(gen)
        std_accuracy = np.std(scores)* 10 ** 2
        print("precision:",precision)
        print("recall:",recall)
        print("std:",std_dev)

        return s / self.K

    def check_dimentions(self, dim):  # check number of all feature
        if dim == None:
            return len(self.tr_x[0])
        else:
            return dim'''


#naïve BayesUse classifier
'''class Evaluate:
    def __init__(self, tr_x, tr_y):
        self.tr_x = tr_x
        self.tr_y = tr_y
        self.K = 5

    def evaluate(self, gen):
        mask = np.array(gen) > 0
        al_data = self.tr_x[:, np.nonzero(mask)[0]]
        kf = KFold(n_splits=self.K, shuffle=True)
        s = 0
        scores = []
        for tr_ix, te_ix in kf.split(self.tr_x):
            nb_classifier = GaussianNB()  # Gaussian Naive Bayes classifier is used
            nb_classifier.fit(al_data[tr_ix], self.tr_y[tr_ix])
            s += nb_classifier.score(al_data[te_ix], self.tr_y[te_ix])
            score = knn_classifier.score(al_data[te_ix], self.tr_y[te_ix])
            scores.append(score)

        selectfea_count = np.count_nonzero(gen == 1)
        f = 0.9 * s / self.K + 0.1 * (1 - selectfea_count) / len(gen)
        std_accuracy = np.std(scores) * 10 ** 2

        return std_accuracy
        return s / self.K
    def check_dimentions(self, dim):#check number of all feature
        if dim==None:
            return len(self.tr_x[0])
        else:
            return dim'''

'''def random_search(n,dim):
    """
    create genes list
    input:{ n: Number of population, default=20
            dim: Number of dimension
    }
    output:{genes_list → [[0,0,0,1,1,0,1,...]...n]
    }
    """
    gens=[[0 for g in range(dim)] for _ in range(n)]
    for i,gen in enumerate(gens) :
        r=random.randint(1,dim)
        for _r in range(r):
            gen[_r]=1
        random.shuffle(gen)
    return gens'''

#Population initialization
def random_search(n, dim):
    gens = []
    for _ in range(n):
        total_ones = random.randint(1, dim/2)  # Ensure that the total number of 1's does not exceed dim
        ones_in_first_half = total_ones  # Here it's simply split in half, but you can adjust it as needed
        ones_in_second_half = total_ones//2

        # If the number of ones in the first half exceeds the number of possible positions, it is reallocated
        if ones_in_first_half > dim // 2:
            ones_in_first_half = dim // 2
            ones_in_second_half = total_ones - ones_in_first_half

            # Randomly choosing a location
        positions_first_half = random.sample(range(dim // 2), ones_in_first_half)
        positions_second_half = random.sample(range(dim // 2, dim), ones_in_second_half)

        # Generating feature vectors
        gen = [1 if i in positions_first_half + positions_second_half else 0 for i in range(dim)]

        # Print the number of ones in the current vector (if needed)
        count = sum(1 for num in gen if num == 1)
        # print(f"The number of ones in the current vector: {count}")

        gens.append(gen)

        # print("All the generated vectors:", gens)  # If you want to print all vectors
    return gens



'''def random_search(n, dim):
    gens = []
    for _ in range(n):
        k = random.randint(1, 100)  # Choose the number of ones
        positions = random.sample(range(1, dim+1), k)  # k non-repetitive positions are chosen at random
        gen = [1 if i+1 in positions else 0 for i in range(dim)]  # Generating feature vectors
        gens.append(gen)
    return gens'''

'''def Bmove(x, a, v): #(gens,a,v)
    n,dim=len(x),len(x[0])#n is the number of population individuals and dim is the number of features
    v=[[random.random()*v[j][i]+a[i] for i in range(dim)] for j in range(n)]#rand(n,nn).*v+a
    s=[[abs(math.tanh(_v)) for _v in vv ] for vv in v]
    temp=[[1 if rr<ss else 0 for rr,ss in zip(_r,_s)] for _r,_s in zip([[random.random() for i in range(dim)] for j in range(n)],s)]
    x_moving=[[0 if temp[ind][i]==1 else 1  for i in range(len(temp[ind])) ] for ind in range(len(temp))]#find(t==1)
    #xm(moving)=~xm(moving)
    return x_moving,v'''


def Bmove(x, a, v,I,iteration,r1=0.2,r2=0.8):
    n, dim = len(x), len(x[0])
    v = [[random.random() * v[j][i] + a[i] for i in range(dim)] for j in range(n)]
    s = [[abs(math.tanh(_v)) for _v in vv] for vv in v]

    p = [[r1*(1-I[row][col]) + r2*s[row][col] for col in range(len(I[row]))] for row in range(len(I))]
    temp = [[1 if rr < ss else 0 for rr, ss in zip(_r, _s)] for _r, _s in zip([[random.random() for i in range(dim)] for j in range(n)], p)]
    x_moving = [[0 if temp[ind][i] == 1 else 1 for i in range(len(temp[ind]))] for ind in range(len(temp))]
    matrix = np.array(x_moving)
    # Decreasing function of mutation probability
    def mutation_rate(iteration):
        return 0.1*np.exp(iteration-100)

    # Mutation operation
    for i in range(n):
        if random.random() < mutation_rate(iteration):
            mutation_type = random.choices(['swap', 'flip', 'insert'], weights=[0.3, 0.5, 0.2], k=1)[0]
            if mutation_type == 'swap':
                #Swap operation
                pos1, pos2 = random.sample(range(dim), 2)
                matrix[i][pos1], matrix[i][pos2] = matrix[i][pos2], matrix[i][pos1]
                x_moving = matrix.tolist()
            elif mutation_type == 'flip':
                # Flip operation
                pos1, pos2 = random.sample(range(dim), 2)
                start = min(pos1, pos2)
                end = max(pos1, pos2)
                matrix[i][start:end + 1] = [1 if elem == 0 else 0 for elem in matrix[i][start:end + 1]]
                x_moving = matrix.tolist()
            elif mutation_type == 'insert':
                # Insert operation
                pos1, pos2 = random.sample(range(dim-1), 2)
                start = min(pos1, pos2)
                end = max(pos1, pos2)
                element = matrix[i][start]
                matrix[i][start:end] = x_moving[i][start + 1:end + 1]
                matrix[i][end] = element
                x_moving = matrix.tolist()
                #x_moving[i].insert(end + 1, element)
                #x_moving[i] = x_moving[i][:start-1] + x_moving[i][start + 1:end + 1]
    return x_moving, v
def mc(fit, min_f):
    fmax=max(fit)
    fmin=min(fit)
    #print("fmax:",fmax)
    #print("fmin:",fmin)
    fmean=np.mean(fit)
    i,n=1,len(fit)

    if fmax==fmin:
        m=[1 for i in range(n)]#once(n,1)
    else:
        if min_f==1:
            best=fmin
            worst=fmax
        else:
            best=fmax
            worst=fmin
        m=[(f-worst)/((best-worst)+sys.float_info.epsilon) for f in fit]
    mm=[_m/sum(m) for _m in m]
    #print("Mass：",mm)
    return mm

def BGc(itertion, max_iter,n,g0):
    g=g0*(1/(1 + np.exp(10*(itertion-max_iter/2)/n)))
    #print("g:",g)
    return g

def BGf(m, x, G, Rp, EC, itertion, max_iter):
    n,dim=len(x),len(x[0])
    final_per=2#In the last iteration, only 2 percent of agents apply force to the others
    if EC == 1:
        kbest=final_per+(1-itertion/max_iter)*(100-final_per)
        kbest=round(n*kbest/100)
    else:
        kbest=n
    mm=np.array(m)
    am=[np.argsort(mm)[::-1][i] for i in range(len(mm))]#:
    ds=sorted(am,reverse=True)
    E = [0 for i in range(dim)]  # zero(1,dim)
    for i in range(len(x)):
        if i < 0 or i >= len(x):
            print("Invalid index:", i)

    # Check the length of the sublist
    for i in range(len(x)):
        if len(x[i]) != dim:
            print("Invalid sublist length at index", i)
            print(len(x[i]))

    for i in range(n):
        #E=[0 for i in range(dim)]#zero(1,dim)
        for ii in range(kbest):
            j=ds[ii]
            if j != i:
                R=sum([1 for xi,xj in zip(x[i],x[j]) if xi!=xj])#hammimng dist
                R=R/dim
                for k in range(dim):
                    E[k]=E[k]+random.random()* m[j] *( (x[j][k]-x[i][k]) / (R**Rp+1/dim) )
            else:
                pass

    a=[e*G for e in E]
    return a, kbest

def BGSA(Eval_Func=Evaluate, n=100, m_i=20, dim=None, minf=0, prog=True, EC=1, Rp=1, f_ind=25,L=5,g0=100):
    """
    input:{ Eval_Func: Evaluate_Function, type is class
            n: Number of population, default=100
            m_i: Number of max iteration, default=300
            minf: minimazation flag, default=0, 0=maximization, 1=minimazation
            dim: Number of feature, default=None
            prog: Do you want to use a progress bar?, default=False
            EC: Elite Check, default=1
            Rp: Value between mass, default=1
            f_ind: Value of kbest, default=25
            }

    output:{Best value: type float 0.967
            Best position: type list(int) [1,0,0,1,.....]
            Nunber of 1s in best position: type int [0,1,1,0,1] → 3
            }
    """
    estimate=Eval_Func.evaluate
    if dim==None:
        dim=Eval_Func.check_dimentions(dim)

    #best_bin='0'*int(dim)
    fbest=float("-inf") if minf == 0 else float("inf")
    #best_val=float("-inf") if minf == 0 else float("inf")
    #EC=1
    #Rp=1
    #f_ind=25#24: max-ones, 25: royal-road(王道)
    #minf=minf#0#1:mini,0:maximization
    gens_dic={tuple([0]*dim):float("-inf") if minf == 0 else float("inf")}
    #flag=dr#False
    gens=random_search(n,dim)#[[random.choice([0,1]) for _ in range(dim)] for i in range(n)]
    q = len(gens)  # Get the population
    dim = len(gens[0])  # Assume that all individuals have the same dimension and take the dimension of the first individual
    print("Shape of gens:", (q, dim))
    #bestc=[]
    #meanc=[]
    I = [[1 for d in range(dim)] for i in range(n)]
    v=[[0 for d in range(dim)] for i in range(n)]
    fit=[-float("inf") if minf == 0 else float("inf") for i in range(n)]
    if prog:
        miter=tqdm(range(m_i)) #Show progress bar
    else:
        miter=range(m_i)

    for it in miter:
        print("第",it,"次迭代：")
        for g_i in range(n):
            if  tuple(gens[g_i]) in gens_dic:
                fit[g_i]=gens_dic[tuple(gens[g_i])]
            else:
                fit[g_i]=estimate(gens[g_i])
                gens_dic[tuple(gens[g_i])]=fit[g_i]

        #print("fit:",fit)
        if it > 1:
            if minf==1:
                pass
                #afit=find(fitness>fitold)#minimazation#find is return index_list
                afit=[ind for ind in range(n) if fit[ind] > fitold[ind]]  #Stores a list of the index of the filtered elements that meet the criteria
            else:
                #afit=find(fittness<fitold)#max#
                afit=[ind for ind in range(n) if fit[ind] < fitold[ind]]

            if len(afit)!=0:
                for ind in afit:
                    gens[ind]=gensold[ind]
                    fit[ind]=fitold[ind]

        if minf == 1:
            best=min(fit)#min
            best_ind=fit.index(min(fit))
        else:
            best=max(fit)#max
            best_ind=fit.index(max(fit))
        if it==1:
            fbest=best
            lbest=gens[best_ind]
        if minf==1:
            if best<fbest:
                fbest=best
                lbest=gens[best_ind]
        else:
            if best>fbest:
                fbest=best
                lbest=gens[best_ind]

        #bestc=fbest
        #meanc=np.mean(fit)

        m=mc(fit,minf)
        g=BGc(it,m_i,n,g0)
        a,kbest=BGf(m,gens,g,Rp,EC,it,m_i)

        gensold=dc(gens)
        fitold=dc(fit)
        #print("gens:", gens)

        I = [[I[i][j] - 1 / L if gens[i][j] == gensold[i][j] else 1 for j in range(len(gens[i]))] for i in range(len(gens))]
        gens,v=Bmove(gens,a,v,I,it)
        print("fmax:", max(fit))
        print("fmin:",min(fit))
    return fbest,lbest,lbest.count(1)
if __name__ == "__main__":
    # Read data and labels from CSV files
    data = pd.read_csv("Gene file path",header=0,index_col=0)
    #data= data.drop(data.columns[0], axis=1)
    labels = pd.read_csv("Tag file path",usecols=[1], header=0,encoding='utf-8')

    # Extract data as numpy arrays
    tr_x = data.values
    tr_y = labels.values.flatten()
    print(tr_x,tr_x.shape)
    print(tr_y, tr_y.shape)
    # Create an instance of the Evaluate class
    #evaluate_instance = Evaluate().evaluate()

    # Run BGSA algorithm for feature selection
    best_val, best_pos, num_ones = BGSA(Eval_Func=Evaluate(tr_x, tr_y))

    # Convert the selected features to a DataFrame
    selected_features = pd.DataFrame(
        data=best_pos,
        index=data.columns,  # Assuming the first row contains feature names
        columns=["Selected"],
    )
    count = selected_features["Selected"].value_counts()[1]

    print("Number of features：",count)
    # Save the selected features to a new CSV file
    selected_features.to_csv("Storage path")


