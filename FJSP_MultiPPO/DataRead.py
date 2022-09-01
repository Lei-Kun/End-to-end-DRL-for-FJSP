# import numpy as np


def getdata(filename='./FJSSPinstances/0_BehnkeGeiger/Behnke1.fjs'):

    # filename='./FJSSPinstances/1_Brandimarte/BrandimarteMk1.fjs'
    # filename='./FJSSPinstances/0_BehnkeGeiger/Behnke1.fjs'
    # filename='./FJSSPinstances/2a_Hurink_sdata/HurinkSdata1.fjs'
    f=open(filename,'r')
    line=f.readline()

    line_data=line.split()

    numbers_float=list(map(float,line_data))

    n=int(numbers_float[0])
    m=int(numbers_float[1])
    average_num_machine=numbers_float[2]
    #print(n)
    #print(m)
    operations_machines={}#the available machines for each operation of jobs
    operations_times={}#the processing time for each operation of jobs


    # jobs=[[]for i in range(n)]
    numonJobs=[]
    # print(jobs)
    for i in range(n):
        line=f.readline()
        line_data=line.split()
        numbers_float = list(map(int, line_data))
        operation_num=int(numbers_float[0])
        numonJobs.append(operation_num)
        # operations=[[] for j in range(operation_num)]
        jj=1
        j=0
        while jj<len(numbers_float):
            o_num=int(numbers_float[jj])
            job_op=[]
            job_machines=[]
            job_processingtime=[]
            for kk in range(0,o_num*2,2):
                # job_op.append(numbers_float[jj+kk+1])
                job_machines.append(numbers_float[jj+kk+1])
                job_processingtime.append(numbers_float[jj+kk+1+1])
            # operations[j]=job_op
            operations_machines[(i+1,j+1)]=job_machines
            for l in range(len(job_machines)):
                operations_times[(i + 1, j + 1,job_machines[l])] = job_processingtime[l]

            j+=1
            jj+=o_num*2+1
    f.close()  # close the file
        # jobs[i]=operations

# print(operations_machines)
    # print(operations_times)
    # print(numonJobs)
    J=list(range(1,n+1)) #define the index of jobs
    M=list(range(1,m+1)) #define the index of machines
    OJ={}
    for j in range(n):
        OJ[(J[j])]=list(range(1,numonJobs[j]+1))

    #define large_M
    largeM=0
    for job in J:
        for op in OJ[(job)]:
            protimemax=0
            for l in operations_machines[(job,op)]:
                if protimemax<operations_times[(job,op,l)]:
                    protimemax=operations_times[(job,op,l)]
            largeM+=protimemax


    Data={
        'n':n,
        'm':m,
        'J':J,
        'M':M,
        'OJ':OJ,
        'operations_machines':operations_machines,
        'operations_times':operations_times,
        'largeM':largeM,
    }
    return Data
#data=getdata('./FJSSPinstances/1_Brandimarte/BrandimarteMk6.fjs')
#print(data)
    # print(largeM)

    # print(operations_times)




    # data=[]
    # for line in open(filename,"r"):
    #     line_data=line.split()
    #     numbers_float=map(float,line_data)
    #     print(line_data)
    #     print(type(line_data))
    #     data.append(line)
    # print(data)
    # with open(filename,'r') as f:
    #     my_data=f.readline()
    #     line_data=my_data.split()
    #     numbers_float = list(map(float, line_data))
    #     print(line_data)
    #     print(numbers_float)
        # for line in my_data:
        #     line_data=line.split()
        #     numbers_float=map(float,line_data)
        # print(numbers_float)
