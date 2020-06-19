#枚举便利最优解（验证最优性）与动画演示
# 作者：胡誉闻

import gurobipy as gp
from gurobipy import GRB
import geatpy as ea
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns 
import time
import copy
import csv
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

sita=0.1 #A SMALL NUMBER? MIP??????


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

def aimfun(Phen):
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

#----------------------------------------测试
# y=[1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]
# F_test=[75,75,75, 81,81, 88,88, 99,99, 114,114, 124, 139, 159, 184]
# B_test=[64,64,64, 81,81, 88,88, 99,99, 114,114, 124, 13, 159, 22]
# y=[1, 0]
# F_test=[40,40]
# B_test=[20,20]
# (obj_tmp,qt,aqt,It,IPt,CIt)=calculate_obj(F_test,B_test,y)
# print(It)
# print(Dt)
# output_file_and_pic(F_test,B_test,y)
# exit(0)
#--------------------------------------

print('---------------------FIRST PHASE------------------------')

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
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('AO补充量与线边库存量-下界问题')
    plt.plot(qt_opt,label='补充AO数量')
    plt.plot(It_opt,label='线边库库存')
    plt.xlabel('周计划/周')
    plt.ylabel('AO数量/个')
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


print('---------------------MIDDLE PHASE------------------------')

#NOW USE THIS SOLUTION TO CALCULATE THE ORIGINAL 

#INTRODUCE STOCKOUT COST?

#FIRST WE DO SOME PRECALCULATION


(F_est,B_est)=estimate_parameters(It_opt,qt_opt)

#
combination=[]
obj=[]

for i in range(2**(N-1),2**N):
    tmp_list=decode_combination(i)
    combination.append(tmp_list)

for i in range(len(combination)):
    relation=combination[i]
    (F,B)=decode_relation(F_est,B_est,relation)
    (obj_tmp,qt,aqt,It,IPt,CIt)=calculate_obj(F,B,relation)
    obj.append(obj_tmp)
    if obj_tmp==min(obj):
        best_relation=relation
        best_obj=obj_tmp
        print(relation)
print(best_relation)
print(best_obj)


#animation
def update(i): 
    relation=combination[i]
    fig.suptitle(str(relation))
    (F,B)=decode_relation(F_est,B_est,relation)
    (Ft,Bt)=expand_parameters(F,B)
    (obj_tmp,qt,aqt,It,IPt,CIt)=calculate_obj(F,B,relation)
    axs[0].clear()
    axs[0].plot(Ft,label='AO补充水平')
    axs[0].plot(Bt,label='AO目标水平')
    axs[0].set_title('物流参数')
    axs[0].set_xlabel('时间/天')
    axs[0].set_ylabel('AO数量/个')
    axs[0].legend()

    axs[1].clear()
    axs[1].plot(It,label='线边库AO数量')
    axs[1].plot(qt,label='补充AO数量')
    axs[1].set_title('AO补充量与线边库存量')
    axs[1].set_xlabel('时间/天')
    axs[1].set_ylabel('AO数量/个')
    axs[1].legend()

def draw_animation(n):
    for i in FI:
        axs[0].axvline(x=i-1,color='#d46061',linestyle=':',linewidth=1)
        axs[1].axvline(x=i-1,color='#d46061',linestyle=':',linewidth=1)
    ani = FuncAnimation(fig, update, frames=range(n), interval=1,repeat=False)
    plt.show()

n=len(combination)
fig,axs=plt.subplots(1,2)
axs[0].set_title('物流参数')
axs[0].set_xlabel('时间/天')
axs[0].set_ylabel('AO数量/个')
axs[1].set_title('AO补充量与线边库存量')
axs[1].set_xlabel('时间/天')
axs[1].set_ylabel('AO数量/个')
draw_animation(n)

print('---------------------SOME OUTPUT------------------------')

(third_F,third_B)=decode_relation(F_est,B_est,best_relation)
third_relation=best_relation

best_adjust=adjust_optimal(third_F,third_B,third_relation)
(final_F,final_B,final_relation)=merge_to_parameter(third_F,third_B,third_relation,best_adjust)
output_file_and_pic(final_F,final_B,final_relation)
(obj_tmp_third,qt_third,aqt_third,It_third,IPt_third,CIt_third)=calculate_obj(final_F,final_B,final_relation)
print('obj',obj_tmp_third)
print(final_relation)

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

timeend = time.time() - timestart
print('time', timeend)

