''' Not the best and the fastest of codes, but it works quite fine for an exam purpose. It implements some of the aspects
of the Heston Model as presented in the book of Rouah, The Heston Model and Its Extensions in Matlab and C, but in python
ad with some slight modifications'''
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm
import time
from tqdm import tqdm
from scipy.special import ndtri # for the inverse CDF


start = time.time()

#============= Control Panel =============
call1 = False # call 1 and call 2 are just two example of call pricining
call2 = False
callMC = False # call price for MC comparisions
set1 = False # Oscillatory Behaviour 1
set2 = False # Oscillatory Behaviour 2
set3 = False # Discontinuity Behaviour 1
set4 = False # Discontinuity Behaviour 2
set5 = False # Figure 1.1 (3D)
set6 = False # Heston Little Trap vs Heston
DistributionLogS = False # Effect of rho and sgm on the distribution of logS
diffcall = False # Simple comparision of the prices given by the two model
diffplot = False # Plot of the difference of BSM and Heston call prices
impl_vol = False # Extract the implied volatility through a bisection algorithm

MONTECARLO = False
samplepaths = True
#=========================================

def Heston(phi, S, K, r, q, v0, kappa, sgm, rho, theta, lamb, tau, Trap):
    '''Heston model with the original formulation of th characteristic function (Heston 1993) and the Little Trap formulation (Albrecher 2007). It
    returns, in order, the RealPart of the integrand of the in-the-money probability P1 and P2, the value of the Call, the value of the Put and the Characteristic
    Function.
    
    Trap == 1 for the Little Trap Formulation'''
    
    x = np.log(S)
    a = kappa*theta
    i = np.complex(0,1) 
    b1 = (kappa+lamb-rho*sgm)
    b2 = (kappa+lamb)
    b = np.array([b1,b2])
    
    d1 = np.sqrt((rho*sgm*i*phi - b[0])**2 - sgm**2*(2*0.5*i*phi - phi**2))
    d2 = np.sqrt((rho*sgm*i*phi - b[1])**2 - sgm**2*(-2*0.5*i*phi - phi**2))
    d = np.array((d1,d2))
    
    g1 = (b[0] - rho*sgm*i*phi + d[0]) / (b[0] - rho*sgm*i*phi - d[0])
    g2 = (b[1] - rho*sgm*i*phi + d[1]) / (b[1] - rho*sgm*i*phi - d[1])
    g = np.array((g1,g2))
    
    if Trap == 0:
        G = (1 - g*np.exp(d*tau)) / (1 - g)
        C1 = (r-q)*i*phi*tau + a/sgm**2 *((b[0] - rho*sgm*i*phi + d[0])*tau - 2*np.log(G[0]))
        C2 = (r-q)*i*phi*tau + a/sgm**2 *((b[1] - rho*sgm*i*phi + d[1])*tau - 2*np.log(G[1]))
        C = np.array((C1,C2))
        
        D1 = (b[0] - rho*sgm*i*phi + d[0])/sgm**2 * ((1 - np.exp(d[0]*tau))/(1 - g[0]*np.exp(d[0]*tau)))
        D2 = (b[1] - rho*sgm*i*phi + d[1])/sgm**2 * ((1 - np.exp(d[1]*tau))/(1 - g[1]*np.exp(d[1]*tau)))
        D = np.array((D1,D2))
    if Trap == 1:
        c = 1/g
        D1 = (b[0] - rho*sgm*i*phi - d[0])/sgm**2 * ((1 - np.exp(-d[0]*tau))/(1 - c[0]*np.exp(-d[0]*tau)))
        D2 = (b[1] - rho*sgm*i*phi - d[1])/sgm**2 * ((1 - np.exp(-d[1]*tau))/(1 - c[1]*np.exp(-d[1]*tau)))
        D = np.array((D1,D2))
        
        G = (1-c*np.exp(-d*tau))/(1-c)
        
        C1 = (r-q)*i*phi*tau + a/sgm**2 *((b[0] - rho*sgm*i*phi - d[0])*tau - 2*np.log(G[0]))
        C2 = (r-q)*i*phi*tau + a/sgm**2 *((b[1] - rho*sgm*i*phi - d[1])*tau - 2*np.log(G[1]))
        C = np.array((C1,C2))
    
    f = np.exp(C + D*v0 + i*phi*x)
    RealPart = np.real(np.exp(-i*phi*np.log(K)) * f/(i*phi)) 
    
    I1 = np.trapz(RealPart[0], dx = 0.001)
    I2 = np.trapz(RealPart[1], dx = 0.001)
    P1 = 0.5 + 1/(math.pi) * I1
    P2 = 0.5 + 1/(math.pi) * I2
    
    Call = S*np.exp(-q*tau)*P1 - K*np.exp(-r*tau)*P2
    Put = Call - S*np.exp(-q*tau) + K*np.exp(-r*tau)
    
    return RealPart, Call, Put, f

def BSMCase(phi, S, K, r, c, sgmBSM, kappa, tau):
    '''Particular Heston case in which it's derived the Black-Scholes-Merton option price.
    To obtain the BSM case from the Heston model it's required that sgm = 0 and theta = v0 = c in the original formulation.
    The BSM variance is sgmBSM = sqrt(v0) = sqrt(c)'''

    i = np.complex(0,1) 

    f2 = np.exp(i*phi*(np.log(S) + (r - 0.5*c)*tau) - 0.5*phi**2*c*tau)
    
    d1 = (np.log(S/K) + (r + sgmBSM**2*0.5)*tau) / (sgmBSM*np.sqrt(tau))
    d2 = d1 - sgmBSM*np.sqrt(tau)
    
    phi_one = norm.cdf(d1) 
    phi_two = norm.cdf(d2)
    
    Call = S*phi_one - K*np.exp(-r*tau)*phi_two
    Put = Call - S + K*np.exp(-r*tau)
    
    return Call, Put, f2

#=========================   Option Price   =========================

if call1:
    phi = np.linspace(0.00001,50,int(50/0.001))
    S = 100; K = 100; r = 0.03; q = 0.0; v0 = 0.05; kappa = 5; sgm = 0.5; rho = -0.8; theta = 0.05; lamb = 0; tau = 0.5
    TAU = np.linspace(0.01,3,100); S = [90, 100, 110]
    C = np.zeros(len(TAU))
    plt.figure()
    for s in S:
        for i, tau in tqdm(zip(range(len(C)),TAU), desc=f'{s}'):
            C[i] = Heston(phi,s,K,r,q,v0,kappa,sgm,rho,theta,lamb,tau,0)[1]
        plt.plot(TAU,C, label=f'S={s}')
    plt.xlabel('Maturity')
    plt.ylabel('Call Price')
    plt.grid(True)
    plt.legend()
    # print(f'Call Price closed form: {C}')
    
if call2:
    phi = np.linspace(0.00001,250,int(150/0.001))
    S = 100; K = 100; r = 0.03; q = 0.0; v0 = 0.05; kappa = 5; sgm = 0.5; rho = -0.8; theta = 0.05; lamb = 0; tau = 0.5
    S = np.linspace(0.01,200,200); TAU = [0, 1, 2]
    C = np.zeros(len(S))
    CBS = np.zeros(len(S)) 
    plt.figure()
    for tau in TAU:
        for i, s in tqdm(zip(range(len(C)),S), desc=f'{tau}'):
            C[i] = Heston(phi,s,K,r,q,v0,kappa,sgm,rho,theta,lamb,tau,0)[1]
            # CBS[i] = BSMCase(phi, s, K, r, v0, sgm, kappa, tau)[0]
        plt.plot(S/K,C/K, label=fr'$\tau$={tau}')
        # plt.plot(S/K,CBS/K, label=fr'$\tau BSM$={tau}')
    plt.xlabel('S/K')
    plt.ylabel('C/K')
    plt.grid(True)
    plt.legend()

if callMC:
    phi = np.linspace(0.00001,50,int(50/0.001))
    S = 100; K = 90; r = 0.03; q = 0.02; v0 = 0.03; kappa = 6.2; sgm = 0.5; rho = -0.7; theta = 0.06; lamb = 0; tau = 0.25
    C = Heston(phi,S,K,r,q,v0,kappa,sgm,rho,theta,lamb,tau,0)[1]
    print(f'Call Price closed form: {C}')

#=========================   Oscillatory Behaviour   =========================
if set1:
    phi = np.linspace(0.00001,100,100000)
    S = 10; K = 10; r = 0; q = 0.0; v0 = 0.07; kappa = 10; sgm = 0.09; rho = -0.9; theta = 0.07; lamb = 0; tau = 1
    
    plt.figure()
    plt.plot(phi,Heston(phi,S,K,r,q,v0,kappa,sgm,rho,theta,lamb,tau,1)[0][0], label = '1 year maturity')

if set2:
    phi = np.linspace(0.00001,80,int(80/0.001))
    phi1 = np.linspace(0.00001,250,int(250/0.001))
    PHI = [phi,phi1]
    S = 7; K = 10; r = 0; q = 0.0; v0 = 0.01; kappa = 10; sgm = 0.175; rho = -0.9; theta = 0.01; lamb = 0; tau = 1/52    
    
    plt.plot(phi,Heston(phi,S,K,r,q,v0,kappa,sgm,rho,theta,lamb,tau,1)[0][0], label = '1/52 years maturity')
    plt.legend()
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'Value of the integrand ($f_1$)')
    plt.grid(True)
    
    
    S = np.linspace(0.5,14,300)
    C0 = np.zeros(len(S))
    C1 = np.zeros(len(S))

    plt.figure()
    for i, s in tqdm(zip(range(len(C0)),S)):
        C0[i] = Heston(phi1,s,K,r,q,v0,kappa,sgm,rho,theta,lamb,tau,0)[1]
        C1[i] = Heston(phi,s,K,r,q,v0,kappa,sgm,rho,theta,lamb,tau,0)[1]
    plt.plot(S/K,C0/K, label=rf'$\phi \in$ [0,{int(max(phi1))}]')
    plt.plot(S/K, C1/K, label=rf'$\phi \in$ [0,{int(max(phi))}]')
    plt.grid(True)
    plt.xlabel('S/K')
    plt.ylabel('C/K')
    plt.legend()
    
#=========================   Discontinuity Behaviour   =========================
if set3:
    phi = np.linspace(0.00001,100,int(100/0.001))
    S = 100; K = 100; r = 0.; q = 0.0; v0 = 0.05; kappa = 10; sgm = 0.75; rho = -0.9; theta = 0.05; lamb = 0; tau = 3
    
    S = np.linspace(0.5,200,300)
    C0 = np.zeros(len(S))
    C1 = np.zeros(len(S))
    
    
    plt.figure()
    for i, s in tqdm(zip(range(len(C0)),S)):
        C0[i] = Heston(phi,s,K,r,q,v0,kappa,sgm,rho,theta,lamb,tau,1)[1]
        C1[i] = Heston(phi,s,K,r,q,v0,kappa,sgm,rho,theta,lamb,tau,0)[1]
    plt.plot(S/K,C0/K, label='Little Trap')
    plt.plot(S/K, C1/K, label='Heston Formulation')
    plt.grid(True)
    plt.xlabel('S/K')
    plt.ylabel('C/K')
    plt.legend()
    
    plt.figure()
    plt.plot(phi,Heston(phi,S,K,r,q,v0,kappa,sgm,rho,theta,lamb,tau,0)[0][0], label= '3 years maturity')
    
if set4:
    phi = np.linspace(0.00001,10,10000)
    S = 100; K = 100; r = 0.; q = 0.0; v0 = 0.05; kappa = 10; sgm = 0.09; rho = -0.9; theta = 0.05; lamb = 0; tau = 1
   
    plt.plot(phi,Heston(phi,S,K,r,q,v0,kappa,sgm,rho,theta,lamb,tau,0)[0][0], label = '1 years maturity')
    plt.grid(True)
    plt.legend()
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'Value of the integrand ($f_1$)')
    plt.show()
    
####### Figure 1.1 #######
if set5:
    phi = np.linspace(-60,60,int(100/0.001))
    S = 7; K = 10; r = 0.; q = 0.0; v0 = 0.07; kappa = 10; sgm = 0.3; rho = -0.9; theta = 0.07; lamb = 0

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.linspace(-50,50,1000)
    y = np.linspace(0.01,0.25,1000)
    x, y = np.meshgrid(x, y)
    z = -Heston(x,S,K,r,q,v0,kappa,sgm,rho,theta,lamb,y,1)[0][0]
    surf = ax.plot_surface(x, y*365, z, cmap=cm.seismic, edgecolor='none', antialiased=True, alpha=0.8)
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel('Maturity (days)')
    ax.set_zlabel(r'Value of the integrand ($f_1$)')
    plt.show()

if set6:
    phi = np.linspace(0.00001,15,int(15/0.001))
    S = 100; K = 100; r = 0.; q = 0.0; v0 = 0.0175; kappa = 1.5768; sgm = 0.5751; rho = -0.5711; theta = 0.0398; lamb = 0; tau = 3.5
    
    plt.figure()
    plt.plot(phi,Heston(phi,S,K,r,q,v0,kappa,sgm,rho,theta,lamb,tau,0)[0][0], label= 'Heston Formulation')
    plt.plot(phi,Heston(phi,S,K,r,q,v0,kappa,sgm,rho,theta,lamb,tau,1)[0][0], label= 'Albrecher Formulation')
    plt.grid(True)
    plt.legend()
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'Integrand ($f_1$)')
    plt.show()
    

#=========================   Distribution on lnS_T and Effects of rho and sgm   =========================

if DistributionLogS:
   phi = np.linspace(0.0001,100,int(100/0.1))
   x = np.linspace(4.3,4.92,int(100/0.1))
   S = 100; K = 100; r = 0.; q = 0.0; v0 = 0.01; kappa = 2; theta = 0.01; lamb = 0; tau = 0.5 
   
   i = np.complex(0,1)
   q = [] ; rho = [-0.8, 0., 0.8]; sgm = [0.0001, 0.2, 0.4]
   
   plt.figure()
   for vak in tqdm(rho,desc=f'{rho}'):
       f = Heston(phi,S,K,r,0.0,v0,kappa,0.1,vak,theta,lamb,tau,1)[3][1]
       for val in x:
           tmp = 1/math.pi * np.trapz(np.real(np.exp(-val*phi*i)*f), dx = 0.001)
           q.append(tmp)
       plt.plot(x,q, label = fr'$\rho$ = {vak}')
       q = []
   plt.legend()  
   plt.xlabel(r'$logS_T$')
   plt.title(r'Distribution of $logS_T$ varying the correlation $\rho$')
   plt.grid(True)
   
   q=[]
   
   plt.figure()
   for val in tqdm(sgm,desc=f'{sgm}'):
       f = Heston(phi,S,K,r,0.0,v0,kappa,val,0.,theta,lamb,tau,1)[3][1]
       for vak in x:
           tmp = 1/math.pi * np.trapz(np.real(np.exp(-vak*phi*i)*f), dx = 0.001)
           q.append(tmp)
       plt.plot(x,q, label = fr'$\sigma$ = {val}')   
       q = []
       
   plt.legend()
   plt.xlabel(r'$logS_T$')
   plt.title(r'Distribution of $logS_T$ varying the variance of volatility $\sigma$')
   plt.grid(True)
   

#=========================   Differences in Price PLOT   =========================

if diffcall:
    
    phi = np.linspace(0.00001,100,int(100/0.001))
    S = 100; K = 100; r = 0.03; q = 0.0; c = 0.05; v0 = 0.05; kappa = 5; sgm = 0.000001; rho = 0; theta = 0.05; lamb = 0; tau = 0.5
    
    BSM = BSMCase(phi, S, K, r, c, np.sqrt(c), kappa, tau)
    H = Heston(phi,S,K,r,q,v0,kappa,sgm,rho,theta,lamb,tau,1)
    print(f'\n BSM Call Price --> {BSM[0]} \n Heston Call Price --> {H[1]} \n BSM Put Price --> {BSM[1]} \n Heston Put Price --> {H[2]}')


if diffplot:
    # funziona tutto, ma devo capire perché le volatilità della BSM le devo moltiplicare per sqrt(2) per trovarmi
    phi = np.linspace(0.00001,100,int(100/0.001))
    K = 100; r = 0.; q = 0.0; v0 = 0.01; kappa = 2; theta = 0.01; lamb = 0; tau = 0.5 # Heston
    K = 100; c = 0.01; # BSM
    
    S = np.linspace(70,140,100) ; RHO = [-0.5, 0.5]; SGM = [np.sqrt(2)*0.0710, np.sqrt(2)*0.0704] # la scrivo direttamente a mano per ora
    
    plt.figure()
    for rho, sgm in zip(RHO, SGM):
        
        BSM_Calls = np.array([BSMCase(phi, s, K, r, c, sgm, kappa, tau)[0] for s in S])
        Hest_Calls = np.array([Heston(phi,s,K,r,q,v0,kappa,0.1,rho,theta,lamb,tau,1)[1] for s in S])
    
        plt.plot(S, Hest_Calls - BSM_Calls, label = fr'$\rho$ = {rho}')
    plt.legend()
    plt.grid(True)
    plt.ylabel('Heston-BSM call price')
    plt.xlabel('Spot Price')
    plt.title(r'Heston-BSM: $\rho$ effect')
  
    
    plt.figure()
    for sgm in [0.1,0.2]:
        
        BSM_Calls1 = np.array([BSMCase(phi, s, K, r, c, np.sqrt(2)*0.0707, kappa, tau)[0] for s in S])
        Hest_Calls1 = np.array([Heston(phi,s,K,r,q,v0,kappa,sgm,0,theta,lamb,tau,1)[1] for s in S])
        plt.plot(S, Hest_Calls1 - BSM_Calls1, label = fr'$\sigma$ = {sgm}')
    plt.legend()
    plt.grid(True)
    plt.title(r'Heston-BSM: $\sigma$ effect')
    
    
#=========================    Implied Volatility    ==============================
impl_3D = False
impl_2D = False
if impl_vol:
    
    phi = np.linspace(0.00001,100,int(100/0.001)); 
    S = 100; r = 0.05; q = 0.0; tau = 0.25; kappa = 2; theta = 0.01; lamb = 0; v0 = 0.01; c = 0.01
    K = np.linspace(95,105,10) 
    RHO = [-0.3,0, 0.3]; KAPPA = [1.0, 2.0, 5.0]; SIGMA = [0.35, 0.55, 0.8]; THETA = [0.010, 0.015, 0.020]; V_0 = [0.010, 0.015, 0.020]; TAU = np.linspace(0.2,1,5)
    
    MaxIter = 20 # bisection iteration
    tol = 0.0001 # bisection tolerance
    
    if impl_2D:
        plt.figure()
        for rho in RHO:
            sgm_implied_rho = []
            time.sleep(0.5)
            for k in tqdm(K, desc=f'rho = {rho} '):
                sgm_low = 0.00001
                sgm_high = 1.
                
                call_low = BSMCase(phi, S, k, r, c, sgm_low, kappa, tau)[0]
                call_high = BSMCase(phi, S, k, r, c, sgm_high, kappa, tau)[0]
                    
                call_market = Heston(phi,S,k,r,q,v0,kappa,0.2,rho,theta,lamb,tau,1)[1]
                
                lowCdif = call_market - call_low
                highCdif = call_market - call_high
                
                if lowCdif*highCdif > 0:
                    # sgm_implied.append(-1)
                    print('Out of the loop: Call Market is always greater than Call Low and Call High')
                    break
                
                for i in range(MaxIter):
                    sgm_mid = (sgm_low + sgm_high)/2
                    call_new = BSMCase(phi, S, k, r, c, sgm_mid, kappa, tau)[0]
                    call_mid = call_market - call_new
                    if abs(call_mid) < tol:
                        sgm_implied_rho.append(sgm_mid)
                    if call_mid > 0:
                        sgm_low = sgm_mid
                    if call_mid < 0:
                        sgm_high = sgm_mid
                
            plt.plot(np.linspace(95,105,len(sgm_implied_rho)), sgm_implied_rho, label = fr'$\rho$ = {rho}')
        
        plt.legend()
        plt.grid(True)
        plt.xlabel('Strike Price')
        plt.ylabel('Implied Volatility')
        plt.title(r'Effect of the correlation $\rho$ on the implied volatility')
        plt.show()
        
        
        plt.figure()
        for sgm in SIGMA:
            time.sleep(0.5)
            sgm_implied_sgm = []
            for k in tqdm(K, desc=f'sigma = {sgm} '):
                sgm_low = 0.00001
                sgm_high = 1.
                
                call_low = BSMCase(phi, S, k, r, c, sgm_low, kappa, tau)[0]
                call_high = BSMCase(phi, S, k, r, c, sgm_high, kappa, tau)[0]
                    
                call_market = Heston(phi,S,k,r,q,v0,kappa,sgm,0,theta,lamb,tau,1)[1]
                
                lowCdif = call_market - call_low
                highCdif = call_market - call_high
                
                if lowCdif*highCdif > 0:
                    # sgm_implied.append(-1)
                    print('Out of the loop: Call Market is always greater than Call Low and Call High')
                    break
                
                for i in range(MaxIter):
                    sgm_mid = (sgm_low + sgm_high)/2
                    call_new = BSMCase(phi, S, k, r, c, sgm_mid, kappa, tau)[0]
                    call_mid = call_market - call_new
                    if abs(call_mid) < tol:
                        sgm_implied_sgm.append(sgm_mid)
                    if call_mid > 0:
                        sgm_low = sgm_mid
                    if call_mid < 0:
                        sgm_high = sgm_mid
                
            plt.plot(np.linspace(95,105,len(sgm_implied_sgm)), sgm_implied_sgm, label = fr'$\sigma$ = {sgm}')
        
        plt.legend()
        plt.grid(True)
        plt.xlabel('Strike Price')
        plt.ylabel('Implied Volatility')
        plt.title(r'Effect of the volatility $\sigma$ on the implied volatility')
        plt.show()
        
        
        plt.figure()
        for kappa in KAPPA:
            time.sleep(0.5)
            sgm_implied_kappa = []
            for k in tqdm(K, desc=f'kappa = {kappa} '):
                sgm_low = 0.00001
                sgm_high = 1.
                
                call_low = BSMCase(phi, S, k, r, c, sgm_low, kappa, tau)[0]
                call_high = BSMCase(phi, S, k, r, c, sgm_high, kappa, tau)[0]
                    
                call_market = Heston(phi,S,k,r,q,v0,kappa,0.2,0,theta,lamb,tau,1)[1]
                
                lowCdif = call_market - call_low
                highCdif = call_market - call_high
                
                if lowCdif*highCdif > 0:
                    # sgm_implied.append(-1)
                    print('Out of the loop: Call Market is always greater than Call Low and Call High')
                    break
                
                for i in range(MaxIter):
                    sgm_mid = (sgm_low + sgm_high)/2
                    call_new = BSMCase(phi, S, k, r, c, sgm_mid, kappa, tau)[0]
                    call_mid = call_market - call_new
                    if abs(call_mid) < tol:
                        sgm_implied_kappa.append(sgm_mid)
                    if call_mid > 0:
                        sgm_low = sgm_mid
                    if call_mid < 0:
                        sgm_high = sgm_mid
                
            plt.plot(np.linspace(95,105,len(sgm_implied_kappa)), sgm_implied_kappa, label = fr'$\kappa$ = {kappa}')
        
        plt.legend()
        plt.grid(True)
        plt.xlabel('Strike Price')
        plt.ylabel('Implied Volatility')
        plt.title(r'Effect of the mean reversion speed $\kappa$ on the implied volatility')
        plt.show()
        
        
        plt.figure()
        for theta in THETA:
            time.sleep(0.5)
            sgm_implied_theta = []
            for k in tqdm(K, desc=f'theta = {theta} '):
                sgm_low = 0.00001
                sgm_high = 1.
                
                call_low = BSMCase(phi, S, k, r, c, sgm_low, kappa, tau)[0]
                call_high = BSMCase(phi, S, k, r, c, sgm_high, kappa, tau)[0]
                    
                call_market = Heston(phi,S,k,r,q,v0,kappa,0.2,0,theta,lamb,tau,1)[1]
                
                lowCdif = call_market - call_low
                highCdif = call_market - call_high
                
                if lowCdif*highCdif > 0:
                    # sgm_implied.append(-1)
                    print('Out of the loop: Call Market is always greater than Call Low and Call High')
                    break
                
                for i in range(MaxIter):
                    sgm_mid = (sgm_low + sgm_high)/2
                    call_new = BSMCase(phi, S, k, r, c, sgm_mid, kappa, tau)[0]
                    call_mid = call_market - call_new
                    if abs(call_mid) < tol:
                        sgm_implied_theta.append(sgm_mid)
                    if call_mid > 0:
                        sgm_low = sgm_mid
                    if call_mid < 0:
                        sgm_high = sgm_mid
                
            plt.plot(np.linspace(95,105,len(sgm_implied_theta)), sgm_implied_theta, label = fr'$\theta$ = {theta}')
        
        plt.legend()
        plt.grid(True)
        plt.xlabel('Strike Price')
        plt.ylabel('Implied Volatility')
        plt.title(r'Effect of the long term variance mean $\theta$ on the implied volatility')
        plt.show()

    if impl_3D:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        sgm_implied_3d = np.zeros((len(TAU),len(K)))
        for i in range(len(TAU)):
            time.sleep(0.5)
            
            for j in tqdm(range(len(K)), desc = f'tau: {i} '):
                sgm_low = 0.00001
                sgm_high = 1.
                
                call_low = BSMCase(phi, S, K[j], r, c, sgm_low, kappa, TAU[i])[0]
                call_high = BSMCase(phi, S, K[j], r, c, sgm_high, kappa, TAU[i])[0]
                    
                call_market = Heston(phi,S,K[j],r,q,v0,kappa,0.2,-0.3,theta,lamb,TAU[i],1)[1]
                
                lowCdif = call_market - call_low
                highCdif = call_market - call_high
                
                if lowCdif*highCdif > 0:
                    # sgm_implied.append(-1)
                    print('Out of the loop: Call Market is always greater than Call Low and Call High')
                    break
                
                for l in range(MaxIter):
                    sgm_mid = (sgm_low + sgm_high)/2
                    call_new = BSMCase(phi, S, K[j], r, c, sgm_mid, kappa, TAU[i])[0]
                    call_mid = call_market - call_new
                    if abs(call_mid) < tol:
                        sgm_implied_3d[i,j] = sgm_mid
                        # sgm_implied_3d.append(sgm_mid)
                    if call_mid > 0:
                        sgm_low = sgm_mid
                    if call_mid < 0:
                        sgm_high = sgm_mid
        x = np.linspace(0.2,1,len(sgm_implied_3d[:,0]))
        y = np.linspace(95,105,len(sgm_implied_3d[:,0]))
        z = sgm_implied_3d
        x, y = np.meshgrid(x, y)
        surf = ax.plot_surface(x, y, z, cmap=cm.seismic, edgecolor='none', antialiased=True, alpha=0.8)
        
        
        plt.show()
 
time.sleep(0.5)
# ======================================================================================
# ======================================================================================
# ==========================    MONTECARLO SIMULATION    ==============================
# ======================================================================================
# ======================================================================================
'''The code for the Quadratic Exponential that I wrote is essentially the same as the one you can find in the book of Rouah (see last pages for the link), but I don't
understand why it do not work properly giving me a wrong estimate. Furthermore, this wrong estimate is the same that Rouah's code generate in Matlab, even if in the book he puts a vary good one!! Is he a LIAR?!'''

if MONTECARLO: 
    N = 100000
    K = 90; r = 0.03; q = 0.02; kappa = 6.2; sgm = 0.5; rho = -0.; theta = 0.06; lamb = 0; tau = 0.25
    tspan = np.linspace(0,tau,200)
    dt = tau/len(tspan)

    call_tmp = np.empty(0)
    call_mean_tmp = np.empty(0)
    num_v_negative = 0
    
    Eulero = False
    Milstein = False; correlation_plot = False;
    Quadratic_Exponential = False
    Pathwise = False
    TransformedVolatility = False
    

    
    if Eulero:
        print('Euler ------------------------------------------------------------')
        time.sleep(0.5)
        
        logS_TN = np.zeros(N)
        v_plot = []
        
        num_v_negative = 0
        for i in tqdm(range(N)):
            S_old = 100
            v_old = 0.03
            for t in tspan:
                Zv = np.random.normal(0,1)
                Z = np.random.normal(0,1)
                Zs = rho*Zv + np.sqrt(1-rho**2)*Z # Cholesky decomposition
                v_new = v_old + kappa*(theta-v_old)*dt + sgm*np.sqrt(v_old)*np.sqrt(dt)*Zv
                if v_new <= 0:
                    num_v_negative += 1
                    v_new = np.abs(v_new)
                S_new = S_old*np.exp((r-q-0.5*v_old)*dt + np.sqrt(v_old)*np.sqrt(dt)*Zs)
                v_old = v_new
                S_old = S_new
                v_plot.append(v_new)
            logS_TN[i] = np.log(S_new)
            # call_tmp = np.append(call_tmp,np.exp(-r*tau)*(max(0,(S_new-K))))
            # call_mean_tmp = np.append(call_mean_tmp,np.mean(call_tmp))
        # call = np.mean(call_tmp)
        # print(f'\nCall Price Heston MC Euler: {call}')
        # print(f'Number of negative variances generated: {num_v_negative}')
        # SE = np.sqrt(1/(N-1) * np.sum((call_tmp-call)**2))/np.sqrt(N)
        # z_a2 = -ndtri(0.01/2)
        # print(f'Standard Error: {SE}')
        # print(f'Confidence Interval: {call} +- {z_a2*SE}')  
        # plt.figure()
        # plt.hlines(C,0,len(call_tmp),colors = 'k',linewidth = 1, label = 'Closed form solution')
        # # plt.hlines(call,0,len(call_tmp),colors = 'r',linewidth = 1, label = 'MC solution')
        # plt.hlines(call - z_a2*SE,0,len(call_tmp),colors = 'r',linewidth = 1, linestyles='dashed', alpha = 0.7, label = 'Confidence Interval MC')
        # plt.hlines(call + z_a2*SE,0,len(call_tmp),colors = 'r',linewidth = 1, linestyles = 'dashed', alpha = 0.7)
        # plt.plot(call_tmp, linewidth = 0.5, c = 'k',alpha = 0.4)
        # plt.xlim([0.4*N, 0.6*N])
        # plt.xlabel('Number of iterations')
        # plt.ylabel('ith call price')
        # plt.legend()
        # plt.grid(True)
        
        # plt.figure()
        # plt.plot(C-call_mean_tmp, c = 'k', linewidth = 1, alpha = 0.9)
        # plt.xlabel('Monte Carlo iterations')
        # plt.ylabel('CF price - MC price')
        # plt.grid(True)
        
        print('------------------------------------------------------------------')
        
    print()
    time.sleep(0.5)
    if Milstein:
        print('Milstein ---------------------------------------------------------')
        time.sleep(0.5)
        
        S_plot = [100]
        v_plot = [0.03]
        
        for i in tqdm(range(N)):
            S_old = 100
            v_old = 0.03
            for t in tspan:
                Zv = np.random.normal(0,1)
                Z = np.random.normal(0,1)
                Zs = rho*Zv + np.sqrt(1-rho**2)*Z # Cholesky decomposition
                v_new = (np.sqrt(v_old)+0.5*sgm*np.sqrt(dt)*Zv)**2 + kappa*(theta-v_old)*dt - 0.25*sgm**2*dt
                if v_new <= 0:
                    num_v_negative += 1
                    v_new = np.abs(v_new)
                S_new = S_old*np.exp((r-q-(1/2)*v_old)*dt + np.sqrt(v_old)*np.sqrt(dt)*Zs)
                S_plot.append(S_new)
                v_plot.append(v_new)
                v_old = v_new
                S_old = S_new
            call_tmp = np.append(call_tmp,np.exp(-r*tau)*(max(0,(S_new-K))))
            call_mean_tmp = np.append(call_mean_tmp,np.mean(call_tmp))
        call = np.mean(call_tmp)
        print(f'\nCall Price Heston MC Milstein: {call}')
        print(f'Number of negative variances generated: {num_v_negative}')
        SE = np.sqrt(1/(N-1) * np.sum((call_tmp-call)**2))/np.sqrt(N)
        z_a2 = -ndtri(0.01/2)
        print(f'Standard Error: {SE}')
        print(f'Confidence Interval: {call} +- {z_a2*SE}')
        plt.figure()
        plt.hlines(C,0,len(call_tmp),colors = 'k',linewidth = 1, label = 'Closed form solution')
        # plt.hlines(call,0,len(call_tmp),colors = 'r',linewidth = 1, label = 'MC solution')
        plt.hlines(call - z_a2*SE,0,len(call_tmp),colors = 'r',linewidth = 1, linestyles='dashed', alpha = 0.7, label = 'Confidence Interval MC')
        plt.hlines(call + z_a2*SE,0,len(call_tmp),colors = 'r',linewidth = 1, linestyles = 'dashed', alpha = 0.7)
        plt.plot(call_tmp, linewidth = 0.5, c = 'k',alpha = 0.4)
        plt.xlim([0.4*N, 0.6*N])
        plt.xlabel('Number of iterations')
        plt.ylabel('ith call price')
        plt.legend()
        plt.grid(True)
        
        plt.figure()
        plt.plot(C-call_mean_tmp, c='k',linewidth = 1, alpha = 0.9)
        plt.xlabel('Monte Carlo iterations')
        plt.ylabel('CF price - MC price')
        plt.grid(True)
        
        if correlation_plot:
            fig, ax1 = plt.subplots()
            ax1.plot(S_plot, color = 'k', label = 'Stock Price', alpha = 0.9)
            ax2 = ax1.twinx()
            ax2.plot(v_plot, color = 'r', label = 'Variance', alpha = 0.9)
            fig.legend()

        print('------------------------------------------------------------------')
             
    time.sleep(0.5)
    if Quadratic_Exponential:
        print('Quadratic exponential --------------------------------------------')
        time.sleep(0.5)
        
        phi_c = 1.5 #Anderson(2008)
        gamma1 = 0.5; gamma2 = 0.5
        K0 = -((kappa*rho*theta)/sgm)*dt
        K1 = (kappa*rho/sgm -0.5)*gamma1*dt - rho/sgm; K2 = (kappa*rho/sgm -0.5)*gamma2*dt + rho/sgm
        K3 = (1-rho**2)*gamma1*dt; K4 = (1-rho**2)*gamma2*dt
        A = K2+0.5*K4
        
        l = 0; j = 0
        E = np.exp(-kappa*dt)
        for i in tqdm(range(N)):
            X_old = np.log(100)
            v_old = 0.03
            
            for t in tspan:
                
                m = theta + (v_old-theta)*E
                s2 = (v_old*sgm**2*E)/kappa * (1-E) + ((theta*sgm**2)/(2*kappa))*(1-E)**2
                phi = s2/m**2
                
                Uv = np.random.uniform(0,1)
                # Simulation of the variance process
                if phi <= phi_c:
                    l += 1
                    b = np.sqrt(2/phi-1+np.sqrt(2/phi*(2/phi-1)))
                    a = m/(1+b**2)
                    Zv = ndtri(Uv)
                    v_new = a*(b+Zv)**2
                    if A < 1/(2*a):
                        M = np.exp((A*b**2*a)/(1-2*A*a))/(np.sqrt(1-2*A*a))
                        K0 = -np.log(M)-(K1+0.5*K3)*v_new
                    U = np.random.uniform(0,1)
                    Z = ndtri(U)
                    X_new = X_old + (r-q)*dt + K0 + K1*v_old + K2*v_new + np.sqrt(K3*v_old + K4*v_new)*Z
                    v_old = v_new
                    X_old = X_new
                if phi > phi_c:
                    j += 1
                    p = (phi-1)/(phi+1)
                    beta = (1-p)/m
                    if (Uv >= 0) and (Uv <= p):
                        phi_inv = 0
                    if (Uv > p) and (Uv <= 1):
                        phi_inv = (1/beta)*np.log((1-p)/(1-Uv))
                    v_new = phi_inv 
                    # Martingala Correction
                    if A < beta :
                        M = p + (beta*(1-p))/(beta-A)
                        K0 = -np.log(M)-(K1+0.5*K3)*v_new
                    # Simulation of the stock price process
                    U = np.random.uniform(0,1)
                    Z = ndtri(U)
                    X_new = X_old + (r-q)*dt + K0 + K1*v_old + K2*v_new + np.sqrt(K3*v_old + K4*v_new)*Z
                    v_old = v_new
                    X_old = X_new
            call_tmp = np.append(call_tmp,np.exp(-r*tau)*(max(0,(np.exp(X_new)-K))))
            call_mean_tmp = np.append(call_mean_tmp,np.mean(call_tmp))
        call = np.mean(call_tmp)
        print(f'\nCall Price Heston MC Quadratic Exponential: {call}')
        print('The QE scheme does not produce negative variances')
        plt.figure()
        plt.hlines(C,0,len(call_tmp)/2,colors = 'r',linewidth = 2, label = 'Closed form solution')
        plt.hlines(call,len(call_tmp)/2,len(call_tmp),colors = 'g',linewidth = 2, label = 'MC solution')
        plt.plot(call_tmp, linewidth = 0.5, c = 'k',alpha = 0.3)
        plt.xlim([0.4*N, 0.6*N])
        plt.xlabel('Number of iterations')
        plt.ylabel('ith call price')
        plt.legend()
        plt.grid(True)
        print(l,j)
        
        plt.figure()
        plt.plot(C-call_mean_tmp)
        plt.xlabel('Monte Carlo iterations')
        plt.ylabel('CF price - MC price')
        plt.grid(True)

        SE = np.sqrt(1/(N-1) * np.sum((call_tmp-call)**2))/np.sqrt(N)
        z_a2 = -ndtri(0.01/2)
        print(f'Standard Error: {SE}')
        print(f'Confidence Interval: {call} +- {z_a2*SE}')
        print('------------------------------------------------------------------')


    time.sleep(0.5)
    if Pathwise:
        print('Pathwise Adapted Linearization Quadratic -------------------------')
        time.sleep(0.5)
        
        for i in tqdm(range(N)):
            S_old = 100
            v_old = 0.03
            for t in tspan:
                Zv = np.random.normal(0,1)
                Z = np.random.normal(0,1)
                Zs = rho*Zv + np.sqrt(1-rho**2)*Z # Cholesky decomposition
                
                thetatil = theta - sgm**2/(4*kappa)
                beta = Zv/np.sqrt(dt)
                v_new = v_old + (kappa*(thetatil - v_old) + sgm*beta*np.sqrt(v_old)) * (1 + (sgm*beta - 2*kappa*np.sqrt(v_old))/(4*np.sqrt(v_old))*dt)*dt
                if v_new <= 0:
                    num_v_negative += 1
                    v_new = np.abs(v_new)
                S_new = S_old*np.exp((r-q-(1/2)*v_old)*dt + np.sqrt(v_old)*np.sqrt(dt)*Zs)
                v_old = v_new
                S_old = S_new
            call_tmp = np.append(call_tmp,np.exp(-r*tau)*(max(0,(S_new-K))))
            call_mean_tmp = np.append(call_mean_tmp,np.mean(call_tmp))
        call = np.mean(call_tmp)
        print(f'\nCall Price Heston MC Pathwise: {call}')
        print(f'Number of negative variances generated: {num_v_negative}')
        plt.figure()
        plt.hlines(C,0,len(call_tmp)/2,colors = 'r',linewidth = 2, label = 'Closed form solution')
        plt.hlines(call,len(call_tmp)/2,len(call_tmp),colors = 'g',linewidth = 2, label = 'MC solution')
        plt.plot(call_tmp, linewidth = 0.5, c = 'k',alpha = 0.3)
        plt.xlim([0.4*N, 0.6*N])
        plt.xlabel('Number of iterations')
        plt.ylabel('ith call price')
        plt.legend()
        plt.grid(True)
        
        plt.figure()
        plt.plot(C-call_mean_tmp)
        plt.xlabel('Monte Carlo iterations')
        plt.ylabel('CF price - MC price')
        plt.grid(True)

        SE = np.sqrt(1/(N-1) * np.sum((call_tmp-call)**2))/np.sqrt(N)
        z_a2 = -ndtri(0.01/2)
        print(f'Standard Error: {SE}')
        print(f'Confidence Interval: {call} +- {z_a2*SE}')
        print('------------------------------------------------------------------')        


    time.sleep(0.5)
    if TransformedVolatility:
        print('Transformed Volatility -------------------------------------------')
        time.sleep(0.5)
        
        logS_T = np.empty(0)
        E = np.exp(-kappa*dt)
        E1 = np.exp(-kappa*(dt/2))
        
        for i in tqdm(range(N)):
            S_old = 100
            w_old = np.sqrt(0.03)
            for t in tspan:
                Zv = np.random.normal(0,1)
                Z = np.random.normal(0,1)
                Zs = rho*Zv + np.sqrt(1-rho**2)*Z # Cholesky decomposition
                m1 = theta + (w_old**2-theta)*E
                m2 = sgm**2/(4*kappa)*(1-E)
                beta = np.sqrt(max(0,m1-m2))
                theta_t = (beta-w_old*E1)/(1-E1)
                w_new = w_old + kappa*0.5*(theta_t-w_old)*dt + 0.5*sgm*np.sqrt(dt)*Zv
                if w_new <= 0:
                    num_v_negative += 1
                    w_new = np.abs(w_new)
                S_new = S_old*np.exp((r-q-0.5*w_old**2)*dt + w_old*np.sqrt(dt)*Zs)
                w_old = w_new
                S_old = S_new
            # logS_T = np.append(logS_T,np.log(S_new))
            call_tmp = np.append(call_tmp,np.exp(-r*tau)*(max(0,(S_new-K))))
            call_mean_tmp = np.append(call_mean_tmp,np.mean(call_tmp))
        call = np.mean(call_tmp)
        print(f'\nCall Price Heston MC T.V.: {call}')
        print(f'Number of negative variances generated: {num_v_negative}')
        SE = np.sqrt(1/(N-1) * np.sum((call_tmp-call)**2))/np.sqrt(N)
        z_a2 = -ndtri(0.01/2)
        print(f'Standard Error: {SE}')
        print(f'Confidence Interval: {call} +- {z_a2*SE}')  
        plt.figure()
        plt.hlines(C,0,len(call_tmp),colors = 'k',linewidth = 1, label = 'Closed form solution')
        plt.hlines(call - z_a2*SE,0,len(call_tmp),colors = 'r',linewidth = 1, linestyles='dashed', alpha = 0.7, label = 'Confidence Interval MC')
        plt.hlines(call + z_a2*SE,0,len(call_tmp),colors = 'r',linewidth = 1, linestyles = 'dashed', alpha = 0.7)
        plt.plot(call_tmp, linewidth = 0.5, c = 'k',alpha = 0.4)
        plt.xlim([0.4*N, 0.6*N])
        plt.xlabel('Number of iterations')
        plt.ylabel('ith call price')
        plt.legend()
        plt.grid(True)
        
        plt.figure()
        plt.plot(C-call_mean_tmp, c = 'k', linewidth = 1, alpha = 0.9)
        plt.xlabel('Monte Carlo iterations')
        plt.ylabel('CF price - MC price')
        plt.grid(True)
        print('------------------------------------------------------------------')  
    

if samplepaths:
    N = 50; tau = 2.5
    tspan = np.linspace(0,tau,300)
    dt = tau/len(tspan)
    v = np.zeros(len(tspan))
    S = np.zeros(len(tspan))
    v[0]= 0.03
    S[0] = 100
    r = 0.03
    theta = 0.06; kappa = 6.2; sgm = 0.5; rho = -0.7; q = 0.02
    for n in tqdm(range(N)):
        for i in range(1,len(tspan)):
            Zv = np.random.normal(0,1)
            Z = np.random.normal(0,1)
            Zs = rho*Zv + np.sqrt(1-rho**2)*Z
            v[i] = (np.sqrt(v[i-1])+0.5*sgm*np.sqrt(dt)*Zv)**2 + kappa*(theta-v[i-1])*dt - 0.25*sgm**2*dt
            S[i] = S[i-1]*np.exp((r-q-(1/2)*v[i-1])*dt + np.sqrt(v[i-1])*np.sqrt(dt)*Zs)
        plt.plot(tspan,S, linewidth = 0.6, alpha=0.9)
    plt.xlabel('Time')
    plt.ylabel('S(t)')

plt.show()




end = time.time()
time_elapsed = (end - start)
print()
print('Elapsed time: %.2f seconds' %time_elapsed)
            
    

    
    
