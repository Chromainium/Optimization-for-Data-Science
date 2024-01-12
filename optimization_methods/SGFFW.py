# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:10:17 2023

@author: chels
"""

import numpy as np
import pandas as pd
from Utils import linear_oracle as LO

def SGFFW(x0, T, m, stochastic_approximator, MGR, objfunc):
    """
    Stochastic Gradient Free Frank Wolfe (SGFFW) Algorithm
    (i.e Algorithm 2 from Sahu et al)

    Args:
    - x0: Initial point (numpy array)
    - m: number of directions (positive interger)
    - T: Number of iterations (positive integer)
    - stochastic_approximator: one of 'KWSA', 'I-RDSA', 'RDSA' (string)

    Returns:
    - x_T: 
    """
    Iterations = []
    Query_Count = []
    Overall_Loss = []
    Distortion_Loss = []
    Attack_Loss = []
    Current_Best_Distortion = []
    
    best_delImgAT = x0  # Initialize the best solution
    best_Loss = 1e10
    
    x = x0.copy()
    
    randBatchIdx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), T, replace=True)
    
    for t in range(T):
        gamma_t = 2 / (t + 8)
        # Step 3 and 4: Compute g(xt, yt) and dt based on the chosen stochastic method (KWSA, RDSA, I-RDSA)
        g, d = objfunc.gradient_estimation(x, gamma_t, m, randBatchIdx, stochastic_approximator)
        
        # Step 5: Compute vt (argmin over the convex set C)
        vt = LO(d, np.inf, 0.1)
        
        # Step 6: Update xt        
        x = (1 - gamma_t) * x + gamma_t * vt
        
        if (objfunc.Loss_Overall < best_Loss):
            best_Loss = objfunc.Loss_Overall
            best_delImgAT = x
            
        MGR.logHandler.write('Iteration Index: ' + str(t))
        MGR.logHandler.write(' Query_Count: ' + str(objfunc.query_count))
        MGR.logHandler.write(' Loss_Overall: ' + str(objfunc.Loss_Overall))
        MGR.logHandler.write(' Loss_Distortion: ' + str(objfunc.Loss_L2))
        MGR.logHandler.write(' Loss_Attack: ' + str(objfunc.Loss_Attack))
        MGR.logHandler.write(' Current_Best_Distortion: ' + str(best_Loss))
        MGR.logHandler.write('\n')
        
        Iterations.append(t)
        Query_Count.append(objfunc.query_count)
        Overall_Loss.append(objfunc.Loss_Overall)
        Distortion_Loss.append(objfunc.Loss_L2)
        Attack_Loss.append(objfunc.Loss_Attack)
        Current_Best_Distortion.append(best_Loss)
        
    SGFFW = pd.Dataframe(np.column_stack([Iterations,
                                          Query_Count,
                                          Overall_Loss,
                                          Distortion_Loss,
                                          Attack_Loss,
                                          Current_Best_Distortion]),
                         columns = ['Iteration Index',
                                    'Query_Count',
                                    'Loss_Overall',
                                    'Loss_Distortion',
                                    'Loss_Attack',
                                    'Current_Best_Distortion'])
        
    SGFFW.to_csv('SGFFW Results.csv')
    
    return best_delImgAT