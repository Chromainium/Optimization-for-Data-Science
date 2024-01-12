# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 13:52:10 2023

@author: chels
"""

import numpy as np
import random
import pandas as pd
import time
from Utils import linear_oracle as LO
from sys import exit

def FZCGS(x0, K, q, L, MGR, objfunc):
    """
    Faster Zeroth-Order Conditional Gradient (FZCGS) Algorithm

    Args:
    - x0: Initial point/ image (numpy array)
    - K: Number of iterations (positive integer)
    - q: q is the sqrt of n where n is Number of component functions (positive integer)
    - L: Lipschitz constant (positive integer)
    
    Returns:
    - x_alpha: Randomly chosen point from {xk}
    """
    Iterations = []
    Query_Count = []
    Overall_Loss = []
    Distortion_Loss = []
    Attack_Loss = []
    Current_Best_Distortion = []    

    best_delImgAT = x0  # Initialize the best solution
    best_Loss = 1e10
    
    '''
    fzcgs(x0, q, mu, K, eta, gamma, n):
        Algorithm 2 from Gao et al
            
        Parameters
        ----------
        x0 : TYPE
            DESCRIPTION. Input image
        q : TYPE
            DESCRIPTION. Sampling parameter (size of S2 sample) = sqrt(n)
        mu : TYPE
            DESCRIPTION. Smoothing parameter = 1/ sqrt(d*K)
        K : TYPE
            DESCRIPTION. Number of iterations
        eta : TYPE
            DESCRIPTION. Termination threshold = 1/K
        gamma : TYPE
            DESCRIPTION. hyperparameter = 1/3L
        n : TYPE
            DESCRIPTION. Number of component functions
    ''' 
    x = x0.copy()       #shape of x0 (28, 28, 1)
    x = x.reshape(-1)
    xprev = x.copy()
    v = np.zeros_like(x)
    vprev = v.copy()
    d = len(x) #dimension of x d=28*28=784
    dprime = d // 16 #i.e we use 6.25% of the pixels
    E = np.random.uniform(-1.0, 1.0, size=(d, dprime))
    norms = np.linalg.norm(E,axis=0,keepdims=True)
    I = E/norms
    #I = np.eye(d) #shape of I: (784, 784)
    
    n = q**2 # using q = np.sqrt(n) does not result in an integer
    mu = 1/np.sqrt(d*K)
    eta = 1/K
    gamma = 1/(3*(L/2)) #1/(3*L)
    
    with open('calls.txt', 'w') as f:
        f.write('NUMBER OF CALLS TO THE FUNCTION EVALUATION\n')
        f.close()
        
    with open('call_times.txt', 'w') as f:
        f.write('DURATION OF EACH FUNCTION EVALUATION CALL\n')
        f.close()
    
    fplus_times = []
    fminus_times = []
    deltafx_times = []
    deltafxprev_times = []
        
    for k in range(K):
        
        if (k % q == 0):
            # Sample S1 without replacement
            S1 = np.random.choice(np.arange(0, MGR.parSet['nFunc']), n, replace=False) #when q was 10, n was 100 and nFunc was 10 so sample was larger than population
            
            #S1_tic = time.perf_counter()
            objfunc.evaluate_counter = 0
            
            for j in range(dprime): #only iterate of subset of d
                ej_vec = I[:,j] #basis vector where only j-th element is 1
                
                #print(ej_vec)
                
                f_plus_tic = time.perf_counter()
                f_plus = objfunc.evaluate(x + mu*ej_vec, S1)
                #print("Type of f_plus", type(f_plus))
                f_plus_toc = time.perf_counter()
                f_plus_time = f_plus_toc - f_plus_tic
                fplus_times.append(f_plus_time)
                #print(x + mu*ej_vec)
                #print("F PLUS : ", f_plus)
                
                f_minus_tic = time.perf_counter()
                f_minus = objfunc.evaluate(x - mu*ej_vec, S1)
                f_minus_toc = time.perf_counter()
                f_minus_time = f_minus_toc - f_minus_tic
                fminus_times.append(f_minus_time)
                #print(x - mu*ej_vec)
                #print("F MINUS : ", f_minus)
                
                f_diff = f_plus - f_minus
                #print("F DIFF : ", f_diff)
                #v[j] = (1/dprime) * (1/(2*mu)) * f_diff #estimate of the gradient
                
                v += (1/dprime) * (1/(2*mu)) * f_diff * ej_vec
                #print(v)
                
                with open('call_times.txt', 'a') as f:
                    f.write('Iteration index: ' + str(k))
                    f.write('  j index: ' + str(j))
                    f.write('  Duration of f_plus calc: ' + str(f_plus_time))
                    f.write('  Duration of f_minus calc: ' + str(f_minus_time))
                    f.write('\n')
                f.close()
            
            #S1_toc = time.perf_counter()
            
            S1_calls = objfunc.evaluate_counter
            
            with open('calls.txt', 'a') as f:
                f.write('Iteration index: ' + str(k))
                f.write('  Size of S1: ' + str(len(S1)))
                f.write('  Calls to obj eval in S1: ' + str(S1_calls))
                f.write('\n')
            f.close()
            
        else:
            # Sample S2 with replacement
            S2 = np.random.choice(np.arange(0, MGR.parSet['nFunc']), q, replace=True)
            
            #S2_tic = time.perf_counter()
            objfunc.evaluate_counter = 0
            
            for qidx in S2: #range(q) remember q is the size of sample S2
                i = np.array([qidx])
                delta_f_x = np.zeros_like(x)
                delta_f_xprev = np.zeros_like(x)
                
                for j in range(dprime): #random.sample(range(d), d//2):
                    ej_vec = I[:,j]
                
                    delta_fx_tic = time.perf_counter()
                    #delta_f_x = objfunc.gradient_estimation(x, mu, q, i)
                    #delta_f_x[j] = (1/(2*mu)) * (objfunc.evaluate(x + mu*ej_vec, i) - objfunc.evaluate(x - mu*ej_vec, i))
                    delta_f_x = (1/(2*mu)) * (objfunc.evaluate(x + mu*ej_vec, i) - objfunc.evaluate(x - mu*ej_vec, i))
                    delta_f_x_vec = delta_f_x * ej_vec
                    
                    delta_fx_toc = time.perf_counter()
                    delta_fx_time = delta_fx_toc - delta_fx_tic
                    deltafx_times.append(delta_fx_time)
                    
                    delta_fxprev_tic = time.perf_counter()
                    #delta_f_xprev = objfunc.gradient_estimation(xprev, mu, q, i)
                    #delta_f_xprev[j] = (1/(2*mu)) * (objfunc.evaluate(xprev + mu*ej_vec, i) - objfunc.evaluate(xprev - mu*ej_vec, i))
                    delta_f_xprev = (1/(2*mu)) * (objfunc.evaluate(xprev + mu*ej_vec, i) - objfunc.evaluate(xprev - mu*ej_vec, i))
                    delta_f_xprev_vec = delta_f_xprev * ej_vec
                    
                    delta_fxprev_toc = time.perf_counter()
                    delta_fxprev_time = delta_fxprev_toc - delta_fxprev_tic
                    deltafxprev_times.append(delta_fxprev_time)
                    
                    v += delta_f_x_vec - delta_f_xprev_vec
                    
                    with open('call_times.txt', 'a') as f:
                        f.write('Iteration index: ' + str(k))
                        f.write('  q index: ' + str(q))
                        f.write('  Duration of delta_fx_time calc: ' + str(delta_fx_time))
                        f.write('  Duration of delta_fxprev_time calc: ' + str(delta_fxprev_time))
                        f.write('\n')
                    f.close()
                #v += delta_f_x - delta_f_xprev # + vprev
                
            v = ((1/dprime) * (1/q) * v) + vprev
            
            #S2_toc = time.perf_counter()
            
            S2_calls = objfunc.evaluate_counter
            
            with open('calls.txt', 'a') as f:
                 f.write('Iteration index: ' + str(k))
                 f.write('  Size of S2: ' + str(len(S2)))
                 f.write('  Calls to obj eval in S2: ' + str(S2_calls))
                 f.write('\n')
            f.close()
        
        vprev = v
        xprev = x.copy()

        # Call the conditional gradient function (condg) here
        #condg_tic = time.perf_counter()
        
        x = condg(v, x, gamma, eta, objfunc, k)
        #print("NEW X")
        #print(x)
        #condg_toc = time.perf_counter()
        
        #condg_time = condg_toc - condg_tic
        
        objfunc.evaluate(x, np.array([]), False)
        print('obj func value', objfunc.evaluate(x, np.array([]), False))
      
        # with open('times.txt', 'a') as f:
        #     f.write('Iteration index: ' + str(k))
        #     f.write('  Objective Function value: ' + str(objfunc.Loss_Overall))
        #     f.write('  Conditional Gradient time: ' + str(condg_time))           
        #     f.write('\n')
        # f.close()

        if (k % 10 == 0):
            print('Iteration Index: ', k)
            objfunc.print_current_loss()
            
        

        if (objfunc.Loss_Overall < best_Loss):
            best_Loss = objfunc.Loss_Overall
            best_delImgAT = np.reshape(x, (28,28,1))

        MGR.logHandler.write('Iteration Index: ' + str(k))
        MGR.logHandler.write(' Query_Count: ' + str(objfunc.query_count))
        MGR.logHandler.write(' Loss_Overall: ' + str(objfunc.Loss_Overall))
        MGR.logHandler.write(' Loss_Distortion: ' + str(objfunc.Loss_L2))
        MGR.logHandler.write(' Loss_Attack: ' + str(objfunc.Loss_Attack))
        MGR.logHandler.write(' Current_Best_Distortion: ' + str(best_Loss))
        MGR.logHandler.write('\n')
        
        Iterations.append(k)
        Query_Count.append(objfunc.query_count)
        Overall_Loss.append(objfunc.Loss_Overall)
        Distortion_Loss.append(objfunc.Loss_L2)
        Attack_Loss.append(objfunc.Loss_Attack)
        Current_Best_Distortion.append(best_Loss)
               
    FZCGS = pd.DataFrame(np.column_stack((Iterations,
                                          Query_Count,
                                          Overall_Loss,
                                          Distortion_Loss,
                                          Attack_Loss,
                                          Current_Best_Distortion)),
                                columns = ['Iterations',
                                          'Query Count',
                                          'Overall Loss',
                                          'Distortion Loss',
                                          'Attack Loss',
                                          'Current BEst Distortion'])
    
    FZCGS.to_csv('FZCGS Results.csv')
    
    func_call_times = pd.DataFrame(np.column_stack((fplus_times, 
                                                      fminus_times)),
                                      columns = ['f_plus calc time',
                                                 'f_minus calc time'])
    
    func_call_times.to_csv('Function_Call_Times.csv')
    
    deltafunc_call_times = pd.DataFrame(np.column_stack((deltafx_times,
                                                      deltafxprev_times)),
                                      columns = ['delta_fx calc time',
                                                 'delta_fxprev calc time'])
    
    deltafunc_call_times.to_csv('Delta Function_Call_Times.csv')
    
    print(best_delImgAT[:,:,0])

    return best_delImgAT


def condg(g, u, gamma, eta, objfunc, k):
    '''
    Parameters
    ----------
    g : TYPE
        DESCRIPTION.
    u : TYPE
        DESCRIPTION.
    gamma : TYPE
        DESCRIPTION.
    eta : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    eta = 0.1
    ut = u.copy()
    gt = g.copy()
    alpha = 1
    t = 1

    while True:
        '''
        inner_product: <g, u-x> = <g, u> - <g, x>
        therefore if g* = g + (1/gamma) * (ut - u)
        then <g*, (ut - x)> = <g*, ut> - <g*, x>
        and max x <g*, (ut - x)> = max x <g*, ut> - <g*, x>
        = <g*, ut> - max x <g*, x> = <g*, ut> + min x <g*, x>
        '''
        #g_star = g + (1/gamma) * (ut - u)
        #print("G STAR")
        #print(g_star)
        
        #LO_tic = time.perf_counter()
        
        #v = LO(g_star, np.inf, 0.1)
        v = -LO(gt, np.inf, 1) * 2
        #v = -LO(gt, np.inf, 1) #* 4 #changed 0.1 to 1
        #v = -np.sign(gt) * 2 #when v is negative, dot prod is negative, u never updates
        #print("is the sign of ut and -v the same?", (np.sign(ut) == -np.sign(v)).all())
        v_hat = ut - v
        #print("LINEAR ORACLE V")
        #print(v)
        
        #print("WHAT IS ut : ", ut)
        #print("WHAT IS v : ", v)
        #print("WHAT IS v-hat : ", v_hat)
        #print("WHAT IS the norm squared : ", np.linalg.norm(v_hat)**2)
        
        #LO_toc = time.perf_counter()
        
        #LO_time = LO_toc - LO_tic
        
        # with open('times.txt', 'a') as f:
        #     f.write('Iteration index: ' + str(k))
        #     f.write('  Linear Oracle time: ' + str(LO_time))
        #     f.write('\n')
        # f.close()
        #print("WHAT IS g : ", gt)
        
        V = np.dot(gt, v_hat) #* 4
        #print("WHAT IS the dot product : ", V)
        
        if V <= eta or t == 10000 or alpha < 0.001:
            #print("DOT PRODUCT, V : ", str(V))
            #print(eta)
            #print('t number in condg : ' + str(t))
            #print("final alpha in condg : " + str(alpha))
            return ut
        
        #a_top = np.dot((1/gamma)*(u-ut)-g,(v_hat-ut))
        a_denom = gamma * (np.linalg.norm(v_hat) ** 2) #changed from (1/gamma) to gamma
        #print("gamma : ", gamma)
        #print("np.linalg.norm(v_hat) ** 2 : ",  np.linalg.norm(v_hat) ** 2)
        alpha = min(1, V / a_denom)
        #print("ALPHA", alpha)
        ut = (1 - alpha) * ut + alpha * v
        gt = g + gamma * (ut - u) #(1/gamma)*(u-ut)-g #changed from (1/gamma) to gamma
        t += 1

