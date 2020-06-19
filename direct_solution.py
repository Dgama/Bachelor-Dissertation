#求解器直接求解-全局最优
# 作者：胡誉闻

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import time
import csv

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


#MODEL DEFINITION


#VARIABLE DEFINITION

m=gp.Model("InventoryManagementProblem")

Bt=m.addVars(FI[N-1],lb=0,vtype=GRB.INTEGER,name='Bt') #REORDER LEVEL OF EACH REVIEW CYCLE

Ft=m.addVars(FI[N-1],lb=0,vtype=GRB.INTEGER,name='Ft') #ORDER UP TO LEVEL OF EACH REVIEW CYCLE

Bi=m.addVars(N,lb=0,vtype=GRB.INTEGER,name='Bi') #REORDER LEVEL OF EACH ASSEMBLY CYCLE

Fi=m.addVars(N,lb=0,vtype=GRB.INTEGER,name='Fi') #ORDER UP TO LEVEL OF EACH ASSEMBLY CYCLE

yi=m.addVars(N,vtype=GRB.BINARY,name='yi') #DECISION VARIABLE USED TO DETERMINE WHETER A REVIEW CYCLE t IS A START OF A NEW LOGISITC CYCLE

Xt=m.addVars(FI[N-1],vtype=GRB.BINARY,name='Xt') #DECISION VARIABLE USED TO DETERMINE WHETER A THERE OCCURS REPLENSHMENT IN REVIEW CYCLE t 

qt=m.addVars(FI[N-1],lb=0,vtype=GRB.INTEGER,name='qt') #NUMBER OF REPLENISHMENT OF REVIEW CYCLE t

aqt=m.addVars(FI[N-1],lb=0,vtype=GRB.INTEGER,name='aqt') #ACCUMULATED NUMBER OF TRANSSHIPMENTS

It=m.addVars(FI[N-1],vtype=GRB.INTEGER,name='It') #INVENTORY IN THE STORAGE AREA CLOSE TO THE ASSEMBLY LINE AT THE END OF REVIEW CYCLE t

IPt=m.addVars(FI[N-1],vtype=GRB.INTEGER,name='IPt') #POSTITION INVENTORY IN THE STORAGE AREA CLOSE TO THE ASSEMBLY LINE AT THE END OF REVIEW CYCLE t

J=m.addVar(lb=0,vtype=GRB.INTEGER,name='J') #NUMBER OF LOGISTIC CYCLES

It_abs=m.addVars(FI[N-1],lb=0,name='It_abs') 

z=m.addVars(FI[N-1],vtype=GRB.BINARY,name='z')


#CONSTRAINTS

m.addConstrs(It[i]*A0<=A for i in range(FI[N-1])) #TOTAL AREA OF THE STOCK IN EACH REVIEW CYCLE DOES NOT EXCEEDS THE AREA LIMIT

m.addConstr(aqt[0]==qt[0]) #BEGIN ACCUMULATING TRANSSHIPMENT

for i in range(1,FI[N-1]): #ACCUMULATING TRANSSHIPMENTS EVERY REVIEW CYCLE
    m.addConstr(aqt[i]==aqt[i-1]+qt[i]) 

m.addConstr(IPt[0]==It[0]) #FIXED 0

if fi_l==0: #WHEN LEAD TIME=0

    m.addConstrs(IPt[i]==It[i] for i in range(1,FI[N-1]))

    m.addConstr(It[0]==I0-Dt[0])

    m.addConstrs(It[i]==I0+aqt[i-1]-Dt[i] for i in range(1,FI[N-1]))

else:
    m.addConstrs(IPt[i]==It[i]+aqt[i-1] for i in range(1,fi_l)) #WHEN LEAD TIME>=1, TIME TO STABLIZE

    m.addConstrs(IPt[i]==It[i]+aqt[i-1]-aqt[i-fi_l] for i in range(fi_l,FI[N-1])) #WHEN LEAD TIME>=1 STABLIZED

    m.addConstrs(It[i]==I0-Dt[i] for i in range(fi_l)) #THE FIST FEW CYCLES WITHIN THE DURATION OF LEADTIME DON'T REPLENISH

    m.addConstrs(It[i]==I0+aqt[i-fi_l]-Dt[i] for i in range(fi_l,FI[N-1])) #THE EXACT INVENTORY OF EACH REVIEW CYCLE


for j in range(N):
    for i in range(FI[j]-fi[j],FI[j]):
        m.addConstr(Ft[i]==Fi[j])  #THE ORDER UP TO LEVEL ARE THE SAME FOR THOSE REVIEW CYCLES WITHIN A SAME ASSEMBLY CYCLE
        m.addConstr(Bt[i]==Bi[j])  #THE REORDER LEVEL ARE THE SAME FOR THOSE REVIEW CYCLES WITHIN A SAME ASSEMBLY CYCLE

m.addConstrs((yi[j]==0)>>(Fi[j]==Fi[j-1]) for j in range(1,N))

m.addConstrs((yi[j]==0)>>(Bi[j]==Bi[j-1]) for j in range(1,N))


m.addConstrs(IPt[i]-Bt[i]<=(1-Xt[i])*M for i in range(0,FI[N-1])) #COMBINING WITH NEXT ONE DECIDES WHETHER POSITION INVENTORY FALL BELOWS THE REORDER LEVEL

m.addConstrs(IPt[i]-Bt[i]>=(Xt[i])*-1*M+sita for i in range(FI[N-1]))

m.addConstrs(Ft[i]-qt[i]-IPt[i]<=(1-Xt[i])*M for i in range(FI[N-1])) #TOGETHER WITH NEXT ONE EXPRESSES THAT IF POSITION INVENTORY IS LOWER THAN REORDER LEVEL, THEN ORDER UP TO LEVEL

m.addConstrs(qt[i]+IPt[i]-Ft[i]<=(1-Xt[i])*M for i in range(FI[N-1]))

m.addConstrs(qt[i]<=Xt[i]*M for i in range(FI[N-1])) #TOGETHER WITH FOLLOWING ONE EXPRESSES THAT IF THE POSITION INVENTORY IS HIGHER THAN THE REORDER LEVEL, THEN NO ORDER OCCURS

m.addConstrs(-qt[i]<=Xt[i]*M for i in range (FI[N-1])) 

m.addConstr(yi.sum('*')==J) #THE NUMBER OF LOGISTIC CYCLE EQUALS THE NUMBER OF ASSEMBLY CYCLES WHICH ARE BEGINING OF A LOGISTICS CYCLE

m.addConstr(yi[0]==1) #???need to be constrain? THE FIRST ASSEMBLY CYCLE WILL SURE BE THE BEGINING OF THE FIRST LOGISTIC CYCLE

m.addConstrs(Ft[i]>=Bt[i] for i in range(FI[N-1])) # ???? the use of i and j? THE ORDER UP TO LEVEL IS GREATER THAN THE REORDER LEVEL FOR EACH REVIEW CYCLE

m.addConstrs(Fi[j]>=Bi[j] for j in range(N)) # THE ORDER UP TO LEVEL IS GREATER THAN THE REORDER LEVEL FOR EACH ASSEMBLY CYCLE

m.addConstrs(It[i]<=z[i]*M for i in range(FI[N-1]))

m.addConstrs(It[i]>=(1-z[i])*-1*M for i in range(FI[N-1]))

m.addConstrs(((z[i]==1)>>(It_abs[i]==h*It[i])) for i in range(FI[N-1]))

m.addConstrs(((z[i]==0)>>(It_abs[i]==-p*It[i])) for i in range(FI[N-1]))

#PROBLEM OBJECTIVE

m.setObjective(CJ*(J-1)+It_abs.sum('*')+CT*qt.sum('*'),GRB.MINIMIZE) #MINIMIZE THE TOTAL COST

#SLOVE THE PROBLEM
m.optimize()

def to_period(yi_opt,Fi_opt,Bi_opt,qt_opt,turnover_rate,It_abs_opt,It_opt):
    yi2=np.array(yi_opt)
    yi3=np.argwhere(yi2==1).tolist()
    print(yi3)
    print(FI)
    print(fi)
    print(len(It_opt))
    print(len(qt_opt))
    avg_qt=[]
    avg_T=[]
    avg_turnover=[]
    avg_turnover2=[]
    corres_day=[]
    f_p=[]
    B_p=[]
    batch=[]
    for i in range(len(yi3)):
        code=yi3[i][0]
        print(code)
        
        f_p.append(Fi_opt[code])
        B_p.append(Bi_opt[code])

        if code!=N-1:
            code2=yi3[i+1][0]
            batch.append(code2-code)
            corres_day.append((code2)*period_duration)
            to_sum_qt=qt_opt[FI[code]-fi[code]:FI[code2]+1]
            to_sum_T=T[FI[code]-fi[code]:FI[code2]+1]
            to_sum_turnover=turnover_rate[code:code2+1]

            to_sum_turnover2=It_abs_opt[FI[code]-fi[code]:FI[code2-1]]

            a=np.argwhere(deltaT*(FI[code]-fi[code])<=accumulated_T_rounded2)[0][0]
            b=np.argwhere(deltaT*FI[code+1]<=accumulated_T_rounded2)[0][0]
            avg_T.append(round(sum(T[a:b])/len(T[a:b]),1))
            avg_qt.append(round(sum(to_sum_qt)/len(to_sum_qt)))
            avg_turnover.append(round(sum(to_sum_turnover)/len(to_sum_turnover),1))
            print(FI[code]-fi[code])
            print(FI[code2])
            if code==0:
                I_start=I0
                I_end=It_opt[FI[code2-1]-1]
                avg_turnover2.append(round(2*sum(to_sum_turnover2)/(I_start+I_end),1))
            else:
                I_start=It_opt[FI[code]-fi[code]-1]
                I_end=It_opt[FI[code2-1]-1]
                if I_start+I_end!=0:
                    avg_turnover2.append(round(2*sum(to_sum_turnover2)/(I_start+I_end),1))
                else: avg_turnover2.append(100)
            
        else:
            batch.append(N-code)
            corres_day.append(length)
            to_sum_qt=qt_opt[FI[code]-fi[code]:]
            to_sum_T=T[FI[code]-fi[code]:FI[code]]
            to_sum_turnover=turnover_rate[code:]

            to_sum_turnover2=It_abs_opt[FI[code]-fi[code]:]
            
            a=np.argwhere(deltaT*FI[code]-fi[code]<=accumulated_T_rounded2)[0][0]
            b=np.argwhere(deltaT*FI[code]<=accumulated_T_rounded2)[0][0]
            avg_T.append(round(sum(T[a:b])/len(T[a:b]),1))
            avg_qt.append(round(sum(to_sum_qt)/len(to_sum_qt)))
            avg_turnover.append(round(sum(to_sum_turnover)/len(to_sum_turnover),1))
            avg_turnover2.append(round(2*sum(to_sum_turnover2)/(It_opt[FI[code]-fi[code]-1]+It_opt[FI[N-1]-1]),1))
    return avg_qt,avg_T,avg_turnover,corres_day,f_p,B_p,batch,avg_turnover2


#OUTPUT VALUES AND SOLUTION

if m.status == GRB.status.OPTIMAL:
    # m.write('inventory information.lp') #WRITE INTO FILE
    qt_opt=[]
    aqt_opt=[]
    It_opt=[]
    PIt_opt=[]
    Ft_opt=[]
    Bt_opt=[]
    Fi_opt=[]
    Bi_opt=[]
    It_abs_opt=[]
    turnover_rate=[]
    yi_opt=[]

    for i in range(FI[N-1]):
        qt_opt.append(round(qt[i].x))
        aqt_opt.append(round(aqt[i].x))
        It_opt.append(round(It[i].x))
        PIt_opt.append(round(IPt[i].x))
        Ft_opt.append(round(Ft[i].x))
        Bt_opt.append(round(Bt[i].x))
        It_abs_opt.append(It_abs[i].x)

    for j in range(N):
        Fi_opt.append(round(Fi[j].x))
        Bi_opt.append(round(Bi[j].x))
        yi_opt.append(round(yi[j].x))
    
    turnover_rate.append(sum(It_abs_opt[:FI[0]])*2/(I0+It_opt[FI[0]-1]))
    for j in range(1,N):
        if (It_opt[FI[j-1]-1]+It_opt[FI[j]-1])!=0:
            turnover_rate.append(sum(It_abs_opt[FI[j-1]:FI[j]])*2/(It_opt[FI[j-1]-1]+It_opt[FI[j]-1]))
        else: turnover_rate.append(0)

    # yi2=np.array(yi_opt)
    # yi3=np.argwhere(yi2==1).tolist()

    (avg_qt,avg_T,avg_turnover,corres_day,f_p,B_p,batch,avg_turnover2)=to_period(yi_opt,Fi_opt,Bi_opt,qt_opt,turnover_rate,It_abs_opt,It_opt)
      

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

    # headers=[i+1 for i in range(N)]
    # with open('res{T0}_{Te}_{K}_{L}_{deltaT}_{I0}_{CJ}_{time}.csv'.format(
    #     T0=T0,Te=TE,K=K,L=L,deltaT=deltaT,I0=I0,CJ=CJ,time=time.strftime("%H%M", time.localtime())),'w') as f2:
    #     f_csv=csv.writer(f2)
    #     f_csv.writerow(headers)
    #     f_csv.writerow(FI)
    #     f_csv.writerow(Fi_opt)
    #     f_csv.writerow(Bi_opt)
    #     f_csv.writerow(turnover_rate)


    sns.set()

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['font.size'] = '30'

    plt.figure('system_dynamic')
    plt.title('物流参数')
    plt.xlabel('时间/周')
    plt.ylabel('AO数量/个')
    plt.plot(Bt_opt,label='补充AO水平')
    plt.plot(Ft_opt,label='目标AO水平')
    
    for i in FI:
        plt.axvline(x=i-1,color='#d46061',linestyle=':',linewidth=1)

    plt.legend()
    plt.show()

    plt.title('AO补充量与线边库存量')
    plt.xlabel('时间/周')
    plt.ylabel('AO数量/个')
    plt.plot(qt_opt,label='补充AO数量')
    plt.plot(It_opt,label='线边库AO数量')
    plt.plot(PIt_opt,label='线边+在途AO数量')

    for i in FI:
        plt.axvline(x=i-1,color='#d46061',linestyle=':',linewidth=1)

    plt.legend()

    plt.show()

    plt.figure('库存周转率')
    plt.title('库存周转率')
    plt.plot(avg_turnover,label='周转率')
    plt.xlabel('物流周期')
    plt.legend()
    plt.show()


  
    wuliu_perid=[j+1 for j in range(round(J.x))]
    print(wuliu_perid)
    print(f_p)
    print(B_p)
    print(yi_opt)
    print(batch)
    print(avg_qt)
    print(avg_T)
    print(avg_turnover)
    print(corres_day)
    print(avg_turnover2)
    print('Together {a} logistic cycles with {b} cost'.format(a=J.x,b=m.ObjVal))
    # with open('res{T0}_{Te}_{K}_{L}_{deltaT}_{I0}_{CJ}_{time}.csv'.format(
    #     T0=T0,Te=TE,K=K,L=L,deltaT=deltaT,I0=I0,CJ=CJ,time=time.strftime("%H%M", time.localtime())),'w') as f2:
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