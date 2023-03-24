# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 17:50:37 2023

@author: jdemange
"""

import numpy as np
import scipy as sp


def mMetropolis(X_init,f_init,phi,rho,gamma):
    X=X_init
    fev=f_init              # échantillon initial (état initial de la Chaine de Markov)
    n=np.shape(X)[1]      # dimension de l'échantillon
    Nc=np.shape(X)[0]     # taille de l'échantillon initial
    #phi=phi              # fonction de performance ou de défaillance
    # rho=rho             # proportion d'échantillons défaillants
    # gamma               # seuil de défaillance
    #print(Nc)
    K=int(1/rho) 
    #print(K)         # nombre de sous-échantillons intermédiaires construits
    N=int(K*Nc)           # taille de l'échantillon final
    #print(N)
    #print(X)
    U=np.zeros((N,n))  
    U_fev=np.zeros(N)
    for k in range(K):
        V=np.copy(X[:,:])
        fevv=np.copy(fev)
        for i in range(n):
                ### étape d'exploration de l'espace (avec un noyau de transition de la CM donné par une densité uniforme) :
            uijk=X[:,i]
            vi=uijk + w*2*np.random.random(Nc)-w*np.ones(Nc) 
            #vi=uijk + 4*np.random.random(Nc)-2*np.ones(Nc) 
                ## pour chaque dimension i et chaque échantillon j, on tire une variable uniforme vi 
                ## sur le segment [uijk-1,uijk+1] autour de uijk (vi~unif([uijk-1,uijk+1]))
                
                ### étape d'acceptation/rejet :
            alpha_ijk=sp.stats.norm.pdf(vi)/sp.stats.norm.pdf(uijk) 
            mat=np.ones((Nc,2))
            mat[:,1]=alpha_ijk
            prob=np.min(mat,axis=1)                                     ## probabilité d'acceptation de l'échantillon vi
            u=np.random.rand(Nc)

            V[:,i]=vi*(u<prob)+V[:,i]*(u>=prob)                                 ## on accepte vi avec probabilité "prob", sinon on rejette et on garde uijk               ## nouvel échantillon (de taille 1) en dimension n
            #print(V)
        phii=phi(V)
        matri=np.copy(X[:,:])
        matri[phii>=gamma,:]=V[phii>=gamma,:]
        U[k*Nc:(k+1)*Nc,:]=matri #ici là il faut revoir   
        U_fev[k*Nc:(k+1)*Nc]=fevv*(phii<gamma)+phii*(phii>=gamma)                        
        ## on accepte V s'il est dans la zone de défaillance
                
        X=U[k*Nc:(k+1)*Nc,:]  
        fev=U_fev[k*Nc:(k+1)*Nc]
             ## nouvel état de la CM (échantillon de taille 1 et de dim n)
        #print(k,X,U)
    return U,U_fev                   ## échantillon final de loi (proche de) la normale conditionnée à être dans la région de défaillance


## Subset simulation dans l'espace gaussien standard + loi uniforme pour l'exploration (cf Au,Beck et Bourinet) pour l'estimation d'une probabilité de défaillance
def SubsetSimu(phi,dim,size,rho):     
    n=dim             # dimension de l'espace des entrées
    N=size            # taille de l'échantillon à chaque itération
    #rho=rho          # quantile / probabilité intermédiaire
    t=1               # initialisation du compteur
    m0=np.zeros(n)   
    v0=np.eye(n)      # moyenne et covariance initiale de la loi gaussienne
    VA0=sp.stats.multivariate_normal(mean=m0,cov=v0)
    X=VA0.rvs(size=N)                                 # échantillon gaussien de taille N
    Ech=[X]
    
    f=phi(X)        
    ford=np.sort(f)               # images de l'échantillon rangées dans l'ordre croissant 

    gamma=ford[int((1-rho)*N)]    # (1-rho)quantile 
    #print("gamma :",gamma)
    Gam=[gamma]
    
    Xel0=X[(f>=gamma),:]  
    fev0=f[f>=gamma]                 # sélection des échantillons défaillants par rapport au 1er seuil gamma
    Xel=Xel0[int(np.shape(Xel0)[0]-rho*N):,:]   # sélection d'exactement rho*N échantillons défaillants
    fev=fev0[int(np.shape(Xel0)[0]-rho*N):]
    while gamma<0 and t<10:            ## arrêt de l'algo dès que gamma dépasse le seuil (0) ou après 10 itérations
        t=t+1
        X,f=mMetropolis(Xel,fev,phi,rho,gamma)           
        Ech.append(X)                           # application de l'algorithme de Metropolis-Hastings pour obtenir un échantillon de loi
                                                # gaussienne conditionnée à être dans la région de défaillance
        #print(np.shape(X))
        #f=phi(X)
        ford=np.sort(f)                            # images de l'échantillon rangées dans l'ordre croissant

        gamma=ford[int((1-rho)*N)]    # (1-rho)quantile
        #print("gamma :",gamma)
        Gam.append(gamma)
        
        Xel0=X[(f>=gamma),:] 
        fev0=f[f>=gamma]                       # sélection des échantillons défaillants par rapport au nouveau seuil gamma
        Xel=Xel0[int(np.shape(Xel0)[0]-rho*N):,:] 
        fev=fev0[int(np.shape(Xel0)[0]-rho*N):]
        # sélection d'exactement rho*N échantillons défaillants
        
        
    pm=np.mean(f>=0)               # estimation de la probabilité que le dernier échantillon soit défaillant (pour le vrai seuil)
    P=rho**(t-1)*pm                # estimation de la probabilité finale
    return(P,t,N*t,Gam)#,Ech         # proba, nb d'itérations, budget, écahntillons à chaque itération, seuils intermédiaires


rho=0.1
N=10000
w=1 #paramtre de spread laissé à 1 pour l'oscillo
