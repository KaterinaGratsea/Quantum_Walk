#algorithm for finding the correct coin operators and generating high dimensional maximally entangled states

#the user changes the parameters : "n=5" #number of steps, "f=5" #number of steps (n=f)
#also the minimization parameters: "niter" , "niter_success" and "T"

import numpy as np
import math
from scipy import optimize
global final
import random

metrhths=0 #counts the iterations of the algorithm

def func(z) :  
    global metrhths 
    metrhths += 1
    n=9 #number of steps  - defined by the user
    k=2*n+1 #number of sites at the final state
    
    initial = np.zeros((2*k,1),dtype=complex)
    #initial state localised on one site
    initial[2*n][0]= 1.
    initial[2*n+1][0]= 1.5
    initial/= np.linalg.norm(initial)
    
    Initial = initial
    #print (Initial)
   
    #definition of matrixS==qplate  (= shift operator )

    qplate = np.zeros((2*k,2*k),dtype=complex)
    #(m,up)--> (m-1,down) (m,down)--> (m+1,up)
    
    i=1
    while (i+2) < 2*k :
        qplate[i][i+1] = 1.0
        i += 2
       
    j=1
    while (j+2) < 2*k :
        qplate[j+1][j] = 1.0
        j += 2
        
    matrixS = qplate    
        
    listSt = [] #list where states are saved
    listc = [] #list where 2 by 2 coin operators are saved
    listC = [] #list where 2k by 2k coin operators are saved

    listSt.append (initial)
    
    #Define coin operators with gdt
    
    l = 0 # for corresponding the correct coin parameters at each step n
    for j in range (0,n,+1) : 
        c=np.zeros((2,2),dtype=complex) #coin operator
        #parameters to be found for each coin operator
        theta=z[0+l]
        ksi=z[1+l]
        zeta=z[2+l]

        #coin operators        
        c[0][0]= (math.cos(ksi*math.pi) + math.sin(ksi*math.pi)*1j)*math.cos(theta*math.pi/2)
        c[0][1]= (math.cos(zeta*math.pi) + math.sin(zeta*math.pi)*1j)*math.sin(theta*math.pi/2) 
        c[1][0]= (math.cos(zeta*math.pi) - math.sin(zeta*math.pi)*1j)*math.sin(theta*math.pi/2)         
        c[1][1]= - (math.cos(ksi*math.pi) - math.sin(ksi*math.pi)*1j)*math.cos(theta*math.pi/2)  
        
        listc.append(c)
        matrixC = np.zeros((2*k,2*k),dtype=complex)
         
        for i in range (0,2*k,2):
            matrixC[0+i][0+i] = c[0][0]
            matrixC[1+i][1+i] = c[1][1]
            matrixC[0+i][1+i] = c[0][1]          
            matrixC[1+i][0+i] = c[1][0]   
         
        listC.append (matrixC)    
        
        #find next state, after applying coin and shift operators
        m1 = np.dot(matrixC,initial)
        m2 = np.dot(matrixS,m1)   #next state

        listSt.append (m2)
        initial = m2/np.linalg.norm(m2)
        l += 3 # moving to the next coin parameters
        
    Phi=initial    
    #reshape state in order to do singular value decomposition
    Phi_reshaped =np.zeros((2,k),dtype=complex)
    for i in range(0,2,1):
        q=0
        for j in range(0,k,1): 
            Phi_reshaped[i][j] = Phi[i+q][0]
            q +=2

    
    psiA, l, psiB = np.linalg.svd(Phi_reshaped,full_matrices=1) #decomposition of initial matrix
    #print ("l",l)
    
    NORM=0.0
    p=1.0 # p has to be larger or equal than 1 for the algorithm to work
    
    for i in range(2) :
         NORM = NORM + math.pow(l[i],p) 

    NORM = math.pow(NORM,1./p)

    #save results    
    with open('test_9_sites_only_NORM.txt', 'a+') as f:
        print (metrhths,",",NORM,file=f)
    f.close()
    
    #details of the state
    if (-NORM+math.sqrt(2)<0.000001) :
        f = open("test_9_sites.txt","a+")
        f.write("initial")
        f.close()
        with open('test_9_sites.txt', 'a+') as f:
            print (Initial,file=f)
        f.close()
        
        f = open("test_9_sites.txt","a+")
        f.write("l,NORM")
        f.close()
        with open('test_9_sites.txt', 'a+') as f:
            print (l,NORM,file=f)
        f.close()
    
        f = open("test_9_sites.txt","a+")
        f.write("z")
        f.close()
        with open('test_9_sites.txt', 'a+') as f:
            print (z,file=f)
        f.close()
        
        
        f = open("test_9_sites.txt","a+")
        f.write("listc")
        f.close()
        with open('test_9_sites.txt', 'a+') as f:
            print (listc,file=f)
        f.close()
        
        
        f = open("test_9_sites.txt","a+")
        f.write("Phi")
        f.close()
        with open('test_9_sites.txt', 'a+') as f:
            print (Phi,file=f)
        f.close()
        
        '''f = open("test_9_sites.txt","a+")
        f.write("Phi_reshaped")
        f.close()
        with open('test_9_sites.txt', 'a+') as f:
            print (Phi_reshaped,file=f)
        f.close() '''   
    
    return (-NORM+ math.sqrt(2))

f=9 #number of steps - defined by the user

#generate the intial random coin parameters 
my_randoms=[]
for i in range (3*f):
    my_randoms.append(random.randrange(1,10,1))

print (my_randoms)    
    
initial_coin_parameters=my_randoms

#minimazation parameters
minimizer_kwargs = {"method": "BFGS"}
#niter,niter_success,T - defined by the user
ret = optimize.basinhopping(func,initial_coin_parameters, minimizer_kwargs=minimizer_kwargs, niter=10, T=1.0, disp = True, niter_success=2 ) # 
 
#creates the 2 by 2 correct coin operators 
l=0
listc=[]
for j in range (0,f,+1) : 
        print ("j",j)
        c=np.zeros((2,2),dtype=complex)
        theta=ret.x[0+l]
        ksi=ret.x[1+l]
        zeta=ret.x[2+l]
        
        c[0][0]= (math.cos(ksi*math.pi) + math.sin(ksi*math.pi)*1j)*math.cos(theta*math.pi/2)
        c[0][1]= (math.cos(zeta*math.pi) + math.sin(zeta*math.pi)*1j)*math.sin(theta*math.pi/2) 
        c[1][0]= (math.cos(zeta*math.pi) - math.sin(zeta*math.pi)*1j)*math.sin(theta*math.pi/2)         
        c[1][1]= - (math.cos(ksi*math.pi) - math.sin(ksi*math.pi)*1j)*math.cos(theta*math.pi/2)  
        
        listc.append(c)
        l+=3
