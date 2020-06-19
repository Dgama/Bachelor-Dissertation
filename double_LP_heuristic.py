# 启发式-双规划算法
# 作者：胡誉闻

import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns 
import time
import copy
import numpy as np
timestart = time.time()
sns.set()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] =False

T0=14 #INITIAL ASSEMBLY CYCLE TIME

TE=5 #TARGET ASSEMBLY CYCLE TIME

length=900

D=159 #AO_CONSUMPTION PER PLANE

L=0 #LEAD TIME OF TRANSHIPMENT

deltaT=6 #REVIEW CYCLE ???0.58

period_duration=60

I0=200 #INITIAL INVENTORY

h=0.2 #HOLDING COST PER AO PER REVIEW CYCLE

p=1

CT=0.2 #TRANSPORTATION COST PER AO

CJ=20 #COST OF CHANGING LOGISTIC PLAN A TIME

A0=0.375 #AREA USED PER A0

A=75 #AVAILABLE AREA OF ONE WORK STATION-----RELAX

M=100000 #ENOUGH LARGE NUMBER

sita=0.1 #A SMALL NUMBER


def SF_parameter_generation(T0,TE,length,D,L,deltaT,period_duration):
    fi_l=int(L/deltaT)

    N_origin=int(2*length/(T0+TE))

    K=(T0-TE)/N_origin #LEARNING SPEED

    T=[T0-K*i for i in range(N_origin)]

    consumption_rate=[D/(T0-K*i) for i in range(N_origin)]

    length_truncated=int(sum(T))#转化时间取整

    accumulated_T_rounded=[int(sum(T[:i])) for i in range(1,len(T)+1)]

    accumulated_T_rounded2=np.array(accumulated_T_rounded) 

    Dt_origin=[]

    for i in range(0,N_origin):
        if i==0:
            Dt_origin+=[consumption_rate[i]]*accumulated_T_rounded[i]
            tmp=sum(T[:i+1])-accumulated_T_rounded[i]
            Dt_origin.append(tmp*consumption_rate[i]+(1-tmp)*consumption_rate[i+1])
        elif i==N_origin-1:
            Dt_origin+=[consumption_rate[i]]*(accumulated_T_rounded[i]-len(Dt_origin))
        else:
            Dt_origin+=[consumption_rate[i]]*(accumulated_T_rounded[i]-len(Dt_origin))
            tmp=sum(T[:i+1])-accumulated_T_rounded[i]
            Dt_origin.append(tmp*consumption_rate[i]+(1-tmp)*consumption_rate[i+1])

    Dt_rouneded=[]

    for i in range(len(Dt_origin)):
        Dt_rouneded.append(int(sum(Dt_origin[:i+1])))

    num_asm_cycle=int(len(Dt_rouneded)/deltaT) #number of assembly cycles

    Dt=[Dt_rouneded[(i+1)*deltaT-1] for i in range(num_asm_cycle)] #consumption of assembly cycle

    if (num_asm_cycle*deltaT)%period_duration!=0:
        N=int(num_asm_cycle*deltaT/period_duration)+1
    else: N=int(num_asm_cycle*deltaT/period_duration)

    FI=[]
    fi=[]

    for i in range(0,N):
        if i==0:
            FI.append(int(period_duration/deltaT))
            fi.append(FI[i])
        elif i==N-1:
            FI.append(num_asm_cycle)
            fi.append(FI[N-1]-FI[N-2])
        else:
            FI.append(int((i+1)*period_duration/deltaT))
            fi.append(FI[i]-FI[i-1])

    return N,fi_l,Dt,fi,FI,accumulated_T_rounded2,T

def estimate_parameters(It_opt,qt_opt):
    F_est=[[0]*(N-i) for i in range(N)] #TO STORE THE ESTIMATION OF F IN DIFFERENT COMBINATION OF ASSEMBLY CYCLES

    B_est=[[0]*(N-i) for i in range(N)] #TO STORE THE ESTIMATION OF B IN DIFFERENT COMBINATION OF ASSEMBLY CYCLES

    IPt_est=It_opt

    for i in range(fi_l,FI[N-1]):
        if i!=0:
            IPt_est[i]=IPt_est[i]+sum(qt_opt[i-fi_l+1:i])
        else: pass

    for i in range(N): #CALCULATE THE EXACT F_est and B_est VALUE
        for j in range(N-i):
            B_est[i][j]=int(sum(IPt_est[FI[i]-fi[i]:FI[i+j]])/(FI[i+j]-FI[i]+fi[i]))
            F_temp=[]
            B_temp=[]
            for k in range(FI[i]-fi[i],FI[i+j]):
                    if qt_opt[k]<=0: pass 
                    else:
                        F_temp.append(qt_opt[k])
                        B_temp.append(IPt_est[k])
                    F_temp.sort()
                    if len(B_temp)==0:
                        B_est[i][j]=0
                    else: B_est[i][j]=int(sum(B_temp)/len(B_temp))
            if len(F_temp)==0: 
                B_est[i][j]=0
                F_est[i][j]=B_est[i][j]
            else:
                F_est[i][j]=round(sum(F_temp)/len(F_temp))
                B_est[i][j]=min(F_est[i][j],max(round(sum(B_temp)/len(B_temp)),F_est[i][j]-F_temp[0]))

    return F_est,B_est

def decode_relation(F_est,B_est,period_relation):
    items=period_relation
    F=[0]*N
    B=[0]*N

    for i in range(N):
            if items[i]==1:
                if i!=N-1:
                    for j in range(i+1,N):
                        if items[j]==1:
                            F[i]=F_est[i][j-i-1]
                            B[i]=B_est[i][j-i-1]
                            break
                        else:
                            F[i]=F_est[i][-1]
                            B[i]=B_est[i][-1]
                else:
                    F[i]=F_est[i][0]
                    B[i]=B_est[i][0]
            else: 
                F[i]=F[i-1]
                B[i]=B[i-1]

    return F,B

def expand_parameters(F,B):
    Ft=[0]*FI[N-1]
    Bt=[0]*FI[N-1]
    for i in range(N):
        for k in range(FI[i]-fi[i],FI[i]):
                Ft[k]=F[i]
                Bt[k]=B[i]
    return Ft,Bt

def calculate_obj(F,B,period_relation):

    (Ft_input,Bt_input)=expand_parameters(F,B)

    J=sum(period_relation)

    qt=[0]*FI[N-1]
    aqt=[0]*FI[N-1]
    It=[0]*FI[N-1]
    IPt=[0]*FI[N-1]
    CIt=[0]*FI[N-1]

    #0
    It[0]=I0-Dt[0]
    IPt[0]=It[0]
    if IPt[0]<=Bt_input[0]:
        qt[0]=Ft_input[0]-IPt[0]
    aqt[0]=qt[0] #special

    if fi_l==0:
        for i in range(1,FI[N-1]):
            It[i]=I0-Dt[i]+aqt[i-1]
            IPt[i]=It[i]
            if IPt[i]<=Bt_input[i]:
                qt[i]=Ft_input[i]-IPt[i]
            aqt[i]=aqt[i-1]+qt[i]
    else:
        #1->fi_l-1
        for i in range(1,fi_l):
            It[i]=I0-int(sum(Dt[:i+1]))
            IPt[i]=It[i]+aqt[i-1]#special
            if IPt[i]<=Bt_input[i]:
                qt[i]=Ft_input[i]-IPt[i]
            aqt[i]=aqt[i-1]+qt[i]

        #fi_l-end
        for i in range(fi_l,FI[N-1]):
            It[i]=I0+aqt[i-fi_l]-int(sum(Dt[:i+1]))
            IPt[i]=It[i]+aqt[i-1]-aqt[i-fi_l]
            if IPt[i]<=Bt_input[i]:
                qt[i]=Ft_input[i]-IPt[i]
            aqt[i]=aqt[i-1]+qt[i]

    #CALCULATE THE MONEY
    for i in range(FI[N-1]):
        if It[i]<0: CIt[i]=-p*It[i]
        else: CIt[i]=h*It[i]



    #PROBLEM OBJECTIVE
    obj_tmp=CJ*(J-1)+sum(CIt)+CT*sum(qt)
    return obj_tmp,qt,aqt,It,IPt,CIt


    combination=Phen.tolist()
    obj=[]
    for items in combination:

        (F,B)=decode_relation(F_est,B_est,items)
        
        #CALCULATION

        obj_tmp = calculate_obj(F,B,items)[0]
    
        obj.append(obj_tmp) #MINIMIZE THE TOTAL COST

        fitv=np.vstack(obj)

    return fitv

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def merge_to_parameter(F,B,best_variable,new_relation):

    F_copy=copy.deepcopy(F)
    B_copy=copy.deepcopy(B)
    relation_copy=copy.deepcopy(best_variable)
    for relation in new_relation:
        temp_F=0
        temp_B=0
        for i in range(relation[1],relation[-1]+1):
            relation_copy[i]=0
        for i in relation:
            temp_F+=F[i]
            temp_B+=B[i]
        avg_F=round(temp_F/len(relation))
        avg_B=round(temp_B/len(relation))
        for i in relation:
            F_copy[i]=avg_F
            B_copy[i]=avg_B

    return F_copy,B_copy,relation_copy

def cluster_algorithem(F,B):
    #制作训练集并且训练
    x=[]
    for i in range(N):
        x.append([F[i],B[i]])
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model.fit(x)

    #画图
    plt.title('Hierarchical Clustering Dendrogram')
    plot_dendrogram(model)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
    pairs=model.children_.tolist()
    return pairs

def pairs_to_relations(pairs):

    all_pairs=[]
    child=[]

    for i in range(len(pairs)):
        items=pairs[i]
        pairs_i=[]
        child_i=[]
        temp_pair=[]
        hhh1=items[0]-N
        hhh2=items[1]-N
        if hhh1>=0: 
            temp_pair=temp_pair+all_pairs[hhh1][0][0]
            child_i.append(hhh1)
            child_i=child_i+child[hhh1]
        else:
            temp_pair.append(items[0])
        if hhh2>=0: 
            temp_pair=temp_pair+all_pairs[hhh2][0][0]
            child_i.append(hhh2)
            child_i=child_i+child[hhh2]
        else:
            temp_pair.append(items[1])
        temp_pair.sort()
        pairs_i.append([temp_pair])
        child.append(child_i)
        
        for j in range(i):
            if j not in child[i]:
                pairs_i.append([all_pairs[j][0][0],temp_pair])

        all_pairs.append(pairs_i)

    return all_pairs

def adjust_optimal(F,B,best_variable):

    #获得聚类关系
    pairs=cluster_algorithem(F,B)
    
    #由聚类关系得到所有的调整方案
    decoded_pair=pairs_to_relations(pairs)

    #根据融合计算新的参数以及关系 
    third_Obj=[]

    for relation_i in decoded_pair:
        for new_relation in relation_i:
            (ad_F,ad_B,ad_relation)=merge_to_parameter(F,B,best_variable,new_relation)
            obj_tmp=calculate_obj(ad_F,ad_B,ad_relation)[0]
            third_Obj.append(obj_tmp)
            if obj_tmp==min(third_Obj):
                best_adjust=new_relation

    return best_adjust

def output_file_and_pic(F_output,B_output,relation_pic):
    (Fi_opt,Bi_opt)=expand_parameters(F_output,B_output)
    (obj_opt,qt_opt,aqt_opt,It_opt,PIt_opt,CIt_opt)=calculate_obj(F_output,B_output,relation_pic)
    draw_pic(Fi_opt,Bi_opt,It_opt,PIt_opt,qt_opt)
    print('best',obj_opt)
    print('relation',relation_pic)
    print('J',sum(relation_pic)-1)

def draw_pic(Ft_output,Bt_output,It_output,PIt_output,qt_output):

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] =False

    # up=plt.subplot(211)
    plt.title('物流参数')
    plt.xlabel('时间/天')
    # plt.xlim(0,FI[N-1])
    # plt.ylim(0,120)
    plt.ylabel('AO数量/个')
    plt.plot(Bt_output,label='补充AO水平')
    plt.plot(Ft_output,label='目标AO水平')

    for i in FI:
        plt.axvline(x=i-1,color='#d46061',linestyle=':',linewidth=1)

    plt.legend()
    plt.show()

    # down=plt.subplot(212)
    plt.title('AO补充量与线边库存量')
    plt.xlabel('时间/天')
    plt.ylabel('AO数量/个')
    plt.plot(It_output,label='线边库AO数量')
    plt.plot(PIt_output,label='线边+在途AO数量')
    plt.plot(qt_output,label='补充AO数量')

    for i in FI:
        plt.axvline(x=i-1,color='#d46061',linestyle=':',linewidth=1)

    plt.legend()

    plt.show()

def save_to_txt():
    return 0
    # with open('res{T0}_{Te}_{K}_{L}_{deltaT}_{I0}_{CJ}_{time}.txt'.format(
    #     T0=T0,Te=TE,K=K,L=L,deltaT=deltaT,I0=I0,CJ=CJ,time=time.strftime("%H%M", time.localtime())),'w') as f:
    #     f.write(str(qt_opt)+"\n")
    #     f.write(str(aqt_opt)+"\n")
    #     f.write(str(Bt_opt)+"\n")
    #     f.write(str(Bi_opt)+"\n")
    #     f.write(str(Fi_opt)+"\n")
    #     f.write(str(Ft_opt)+"\n")
    #     f.write(str(It_opt)+"\n")
    #     f.write(str(PIt_opt)+"\n")
    #     f.write(str(J.x))
    # f.close()

def to_period(yi_opt,Fi_opt,Bi_opt,qt_opt,turnover_rate):
    yi2=np.array(yi_opt)
    yi3=np.argwhere(yi2==1).tolist()
    avg_qt=[]
    avg_T=[]
    avg_turnover=[]
    corres_day=[]
    f_p=[]
    B_p=[]
    batch=[]
    for i in range(len(yi3)):
        code=yi3[i][0]
        f_p.append(Fi_opt[code])
        B_p.append(Bi_opt[code])

        if code!=N-1:
            batch.append(yi3[i+1][0]-code)
            corres_day.append((yi3[i+1][0])*period_duration)
            to_sum_qt=qt_opt[FI[code]-fi[code]:FI[yi3[i+1][0]]+1]
            to_sum_T=T[FI[code]-fi[code]:FI[yi3[i+1][0]]+1]
            to_sum_turnover=turnover_rate[code:yi3[i+1][0]+1]
            a=np.argwhere(deltaT*(FI[code]-fi[code])<=accumulated_T_rounded2)[0][0]
            b=np.argwhere(deltaT*FI[code+1]<=accumulated_T_rounded2)[0][0]
            avg_T.append(round(sum(T[a:b])/len(T[a:b]),1))
            avg_qt.append(round(sum(to_sum_qt)/len(to_sum_qt)))
            avg_turnover.append(round(sum(to_sum_turnover)/len(to_sum_turnover),1))
            
        else:
            batch.append(N-code)
            corres_day.append(length)
            to_sum_qt=qt_opt[FI[code]-fi[code]:]
            to_sum_T=T[FI[code]-fi[code]:FI[code]]
            to_sum_turnover=turnover_rate[code:]
            a=np.argwhere(deltaT*FI[code]-fi[code]<=accumulated_T_rounded2)[0][0]
            b=np.argwhere(deltaT*FI[code]<=accumulated_T_rounded2)[0][0]
            avg_T.append(round(sum(T[a:b])/len(T[a:b]),1))
            avg_qt.append(round(sum(to_sum_qt)/len(to_sum_qt)))
            avg_turnover.append(round(sum(to_sum_turnover)/len(to_sum_turnover),1))
    return avg_qt,avg_T,avg_turnover,corres_day,f_p,B_p,batch

def decode_combination(i):
    tmp=i
    tmp_list=[]
    for j in range(N):
        a=int(tmp%2)
        tmp=(tmp-a)/2
        tmp_list.append(a)
    tmp_list.reverse()
    return tmp_list

(N,fi_l,Dt,fi,FI,accumulated_T_rounded2,T)=SF_parameter_generation(T0,TE,length,D,L,deltaT,period_duration)

#MODEL DEFINITION


#VARIABLE DEFINITION

m=gp.Model("first phase")

qt=m.addVars(FI[N-1],lb=0,vtype=GRB.INTEGER,name='qt') #NUMBER OF REPLENISHMENT OF REVIEW CYCLE t

It=m.addVars(FI[N-1],lb=0,vtype=GRB.INTEGER,name='It') #INVENTORY IN THE STORAGE AREA CLOSE TO THE ASSEMBLY LINE AT THE END OF REVIEW CYCLE t

aqt=m.addVars(FI[N-1],lb=0,vtype=GRB.INTEGER,name='aqt') #ACCUMULATED NUMBER OF TRANSSHIPMENTS


#CONSTRAINTS

m.addConstrs(It[i]*A0<=A for i in range(FI[N-1])) #TOTAL AREA OF THE STOCK IN EACH REVIEW CYCLE DOES NOT EXCEEDS THE AREA LIMIT

m.addConstr(aqt[0]==qt[0]) #BEGIN ACCUMULATING TRANSSHIPMENT

for i in range(1,FI[N-1]):
    m.addConstr(aqt[i]==aqt[i-1]+qt[i]) #ACCUMULATING TRANSSHIPMENTS EVERY REVIEW CYCLE

m.addConstrs(It[i]==I0-Dt[i] for i in range(fi_l)) #THE FIST FEW CYCLES WITHIN THE DURATION OF LEADTIME DON'T REPLENISH

m.addConstrs(It[i]==I0+aqt[i-fi_l]-Dt[i] for i in range(fi_l,FI[N-1])) #THE EXACT INVENTORY OF EACH REVIEW CYCLE


#PROBLEM OBJECTIVE

m.setObjective(h*It.sum('*')+CT*qt.sum('*'),GRB.MINIMIZE) #MINIMIZE THE TOTAL COST

#SLOVE THE PROBLEM
m.optimize()

#OUTPUT VALUES AND SOLUTION

if m.status == GRB.status.OPTIMAL:
    # m.write('inventory information.lp') #WRITE INTO FILE
    qt_opt=[]
    aqt_opt=[]
    It_opt=[]

    for i in range(FI[N-1]):
        qt_opt.append(qt[i].x)
        aqt_opt.append(aqt[i].x)
        It_opt.append(It[i].x)

    
    # with open('Res{T0}_{Te}_{K}_{L}_{deltaT}_{I0}_{CJ}_{time}.txt'.format(
    #     T0=T0,Te=TE,K=K,L=L,deltaT=deltaT,I0=I0,CJ=CJ,time=time.strftime("%H%M", time.localtime())),'w') as f:
    #     f.write(str(qt_opt)+"\n")
    #     f.write(str(aqt_opt)+"\n")
    #     f.write(str(It_opt)+"\n")

    # f.close()

    plt.plot(qt_opt,label='Order Quantity')
    plt.plot(It_opt,label='Inventory')
    plt.legend()
    plt.show()

elif m.status != GRB.Status.INFEASIBLE:
    print('Optimization was stopped with status %d' % m.status)
else:
    print('')
    print('Model is infeasible')
    m.computeIIS()
    m.write("model.ilp")
    print("IIS written to file 'model.ilp'")
    exit(0)
if(m.objVal<=0):
    exit(0)

print('SECOND PHASE')

#NOW USE THIS SOLUTION TO CALCULATE THE ORIGINAL 

#FIRST WE DO SOME PRECALCULATION

(F_est,B_est)=estimate_parameters(It_opt,qt_opt)

n=gp.Model('second phase') #SECOND PHASE OF OPTIMIZATION

#VARIABLES

xij=[[(n.addVar(vtype=GRB.BINARY,name='y[{i},{j}]'.format(i=i,j=i+j))) for j in range(N-i)] for i in range(N)] #CREATE INDICATE VARIABLES FOR RELATIONSHIPS BETWEEN ASSEMBLY CYCLES

Bi_2nd=n.addVars(N,lb=0,vtype=GRB.INTEGER,name='Bi_2nd') #REORDER LEVEL OF EACH ASSEMBLY CYCLE

Fi_2nd=n.addVars(N,lb=0,vtype=GRB.INTEGER,name='Fi_2nd') #ORDER UP TO LEVEL OF EACH ASSEMBLY CYCLE

yi_2nd=n.addVars(N,vtype=GRB.BINARY,name='yi_2nd') #DECISION VARIABLE USED TO DETERMINE WHETER A REVIEW CYCLE t IS A START OF A NEW LOGISITC CYCLE

Xt_2nd=n.addVars(FI[N-1],vtype=GRB.BINARY,name='Xt_2nd') #DECISION VARIABLE USED TO DETERMINE WHETER A THERE OCCURS REPLENSHMENT IN REVIEW CYCLE t 

qt_2nd=n.addVars(FI[N-1],lb=0,vtype=GRB.INTEGER,name='qt_2nd') #NUMBER OF REPLENISHMENT OF REVIEW CYCLE t

aqt_2nd=n.addVars(FI[N-1],lb=0,vtype=GRB.INTEGER,name='aqt_2nd') #ACCUMULATED NUMBER OF TRANSSHIPMENTS

It_2nd=n.addVars(FI[N-1],lb=-100000,vtype=GRB.INTEGER,name='It_2nd') #INVENTORY IN THE STORAGE AREA CLOSE TO THE ASSEMBLY LINE AT THE END OF REVIEW CYCLE t

IPt_2nd=n.addVars(FI[N-1],lb=-100000,vtype=GRB.INTEGER,name='IPt_2nd') #POSTITION INVENTORY IN THE STORAGE AREA CLOSE TO THE ASSEMBLY LINE AT THE END OF REVIEW CYCLE t

J_2nd=n.addVar(lb=0,vtype=GRB.INTEGER,name='J_2nd') #NUMBER OF LOGISTIC CYCLES

It_abs_2nd=n.addVars(FI[N-1],lb=0,name='It_abs_2nd') 

z_2nd=n.addVars(FI[N-1],vtype=GRB.BINARY,name='z_2nd')

#CONSTRAINTS

#DECIDING THE EXACT PARAMETER FOR DIFFERENT PERIOD

n.addConstrs(yi_2nd[i]==sum(xij[i][:]) for i in range(N)) #COME OUT RULE FOR EACH PERIOD 

n.addConstrs(sum(xij[j][i-j-1] for j in range(i))==yi_2nd[i] for i in range(1,N)) #COME IN RULE FOR EACH PERIOD (EXCLUDE THE FIRST PERIOD)

n.addConstr(1==yi_2nd[0]) #ASSEMBLY CYCLE 0 HAS TO BE A START, SO THERE IS COME-IN-AND-COME-OUT EFFET

n.addConstrs((yi_2nd[i]==1)>>(Fi_2nd[i]==sum(F_est[i][j]*xij[i][j] for j in range(N-i))) for i in range(N)) #IF A LOGISTIC PERIOD IS FROM i TO j, THEN THE ORDER UP TO LEVEL CAN BE DETERMINED

n.addConstrs((yi_2nd[i]==1)>>(Bi_2nd[i]==sum(B_est[i][j]*xij[i][j] for j in range(N-i))) for i in range(N)) #IF A LOGISTIC PERIOD IS FROM i TO j, THEN THE REORDER LEVEL CAN BE DETERMINED

n.addConstrs(((yi_2nd[i]==0)>>(Fi_2nd[i]==Fi_2nd[i-1])) for i in range(1,N)) #IF THERE IS NO COME-OUT-AND-IN THEN THE PERIOD PARAMETER IS THE SAME AS FORMER PERIOD

n.addConstrs(((yi_2nd[i]==0)>>(Bi_2nd[i]==Bi_2nd[i-1])) for i in range(1,N))

n.addConstr(aqt_2nd[0]==qt_2nd[0]) #BEGIN ACCUMULATING TRANSSHIPMENT

for i in range(1,FI[N-1]): #ACCUMULATING TRANSSHIPMENTS EVERY REVIEW CYCLE
    n.addConstr(aqt_2nd[i]==aqt_2nd[i-1]+qt_2nd[i]) 

n.addConstr(IPt_2nd[0]==It_2nd[0]) #FIXED 0

if fi_l==0: #WHEN LEAD TIME=0

    n.addConstrs(IPt_2nd[i]==It_2nd[i] for i in range(1,FI[N-1]))

    n.addConstr(It_2nd[0]==I0-Dt[0])

    n.addConstrs(It_2nd[i]==I0+aqt_2nd[i-1]-Dt[i] for i in range(1,FI[N-1]))

else:
    n.addConstrs(IPt_2nd[i]==It_2nd[i]+aqt_2nd[i-1] for i in range(1,fi_l)) #WHEN LEAD TIME>=1, TIME TO STABLIZE

    n.addConstrs(IPt_2nd[i]==It_2nd[i]+aqt_2nd[i-1]-aqt_2nd[i-fi_l] for i in range(fi_l,FI[N-1])) #WHEN LEAD TIME>=1 STABLIZED

    n.addConstrs(It_2nd[i]==I0-Dt[i] for i in range(fi_l)) #THE FIST FEW CYCLES WITHIN THE DURATION OF LEADTIME DON'T REPLENISH

    n.addConstrs(It_2nd[i]==I0+aqt_2nd[i-fi_l]-Dt[i] for i in range(fi_l,FI[N-1])) #THE EXACT INVENTORY OF EACH REVIEW CYCLE

for j in range(N):
    for i in range(FI[j]-fi[j],FI[j]):
        n.addConstr(IPt_2nd[i]-Bi_2nd[j]<=(1-Xt_2nd[i])*M) #COMBINING WITH NEXT ONE DECIDES WHETHER POSITION INVENTORY FALL BELOWS THE REORDER LEVEL
        n.addConstr(IPt_2nd[i]-Bi_2nd[j]>=(Xt_2nd[i])*-1*M+sita)
        n.addConstr((Xt_2nd[i]==1)>>(qt_2nd[i]==Fi_2nd[j]-IPt_2nd[i])) #TOGETHER WITH NEXT ONE EXPRESSES THAT IF POSITION INVENTORY IS LOWER THAN REORDER LEVEL, THEN ORDER UP TO LEVEL
        n.addConstr((Xt_2nd[i]==0)>>(qt_2nd[i]==0)) #TOGETHER WITH FOLLOWING ONE EXPRESSES THAT IF THE POSITION INVENTORY IS HIGHER THAN THE REORDER LEVEL, THEN NO ORDER OCCURS

n.addConstr(yi_2nd.sum('*')==J_2nd) #THE NUMBER OF LOGISTIC CYCLE EQUALS THE NUMBER OF ASSEMBLY CYCLES WHICH ARE BEGINING OF A LOGISTICS CYCLE

n.addConstrs(It_2nd[i]<=z_2nd[i]*M for i in range(FI[N-1]))

n.addConstrs(It_2nd[i]>=(1-z_2nd[i])*-1*M for i in range(FI[N-1]))

n.addConstrs(((z_2nd[i]==1)>>(It_abs_2nd[i]==h*It_2nd[i])) for i in range(FI[N-1]))

n.addConstrs(((z_2nd[i]==0)>>(It_abs_2nd[i]==-p*It_2nd[i])) for i in range(FI[N-1]))

n.setObjective(CJ*(J_2nd-1)+It_abs_2nd.sum('*')+CT*qt_2nd.sum('*'),GRB.MINIMIZE)

n.optimize()

if n.status == GRB.status.OPTIMAL:
    yi_2nd_opt=[]
    
    for i in range(N):
        yi_2nd_opt.append(yi_2nd[i].x)

    (third_F,third_B)=decode_relation(F_est,B_est,yi_2nd_opt)

    best_adjust=adjust_optimal(third_F,third_B,yi_2nd_opt)
    (final_F,final_B,final_relation)=merge_to_parameter(third_F,third_B,yi_2nd_opt,best_adjust)
    output_file_and_pic(final_F,final_B,final_relation)

    (obj_tmp_third,qt_third,aqt_third,It_third,IPt_third,CIt_third)=calculate_obj(final_F,final_B,final_relation)

    turnover_rate=[]
    turnover_rate.append(sum(CIt_third[:FI[0]])*2/(I0+It_third[FI[0]-1]))
    for j in range(1,N):
        if (It_third[FI[j-1]-1]+It_third[FI[j]-1])!=0:
            turnover_rate.append(sum(CIt_third[FI[j-1]:FI[j]])*2/(It_third[FI[j-1]-1]+It_third[FI[j]-1]))
        else: turnover_rate.append(0)

    (avg_qt,avg_T,avg_turnover,corres_day,f_p,B_p,batch)=to_period(final_relation,final_F,final_B,qt_third,turnover_rate)
    wuliu_perid=[j+1 for j in range(round(sum(final_relation)))]

    plt.figure('库存周转率')
    plt.title('库存周转率')
    plt.plot(avg_turnover,label='周转率')
    plt.xlabel('物流周期')
    plt.legend()
    plt.show()
    
    # with open('sm{T0}_{Te}_{L}_{deltaT}_{I0}_{CJ}_{time}.csv'.format(
    #     T0=T0,Te=TE,L=L,deltaT=deltaT,I0=I0,CJ=CJ,time=time.strftime("%H%M", time.localtime())),'w') as f2:
    #     f_csv=csv.writer(f2)
    #     f_csv.writerow(wuliu_perid)
    #     f_csv.writerow(corres_day)
    #     f_csv.writerow(batch)
    #     f_csv.writerow(avg_T)
    #     f_csv.writerow(B_p)
    #     f_csv.writerow(f_p)
    #     f_csv.writerow(avg_qt)
    #     f_csv.writerow(avg_turnover)
    # f2.close()

elif n.status != GRB.Status.INFEASIBLE:
    print('Optimization was stopped with status %d' % n.status)
else:
    print('')
    print('Model is infeasible')
    n.computeIIS()
    n.write("model.ilp")
    print("IIS written to file 'model.ilp'")
    exit(0)

timeend = time.time() - timestart
print('time', timeend)