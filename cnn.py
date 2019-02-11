import numpy as np
import math

class Network:
    def __init__(self,N,p,d):
        self.N=N
        self.p=p
        self.d=d
        self.V=np.zeros(N)
        self.r=np.zeros((N,p,d))
        self.J=np.zeros((N,N))
        self.m=np.zeros(p)
        self.g=1.0
        self.h0=0
            
    def BuildMaps(self):
        for i in range(self.N):
            for mu in range(self.p):
                for k in range(self.d):
                    self.r[i][mu][k]=np.random.uniform(0,1)
        return
    
    def ReadMaps(self,R):
        self.r=R
        return
    
    def SetJ(self):
        for i in range(self.N):
            for j in range(self.N):
                if i !=j:
                    for mu in range(self.p):
                        self.J[i][j]=self.J[i][j]+K1cos(self.r[i][mu],self.r[j][mu])
        
        return
    
    def SetV0(self,a1):
        self.V=np.random.uniform(0,1,self.N)
        self.V=self.V*(a1/np.mean(self.V))
        return 
        
    def EvolveDynamics(self,a1,a2):
        Vold=np.zeros(self.N)
        h=np.zeros(self.N)
        converged=False
        it=0
        maxiter=50
        while (not converged and it<maxiter):
            h=np.dot(self.J,self.V)
            self.V=np.asarray(list(map(lambda x: TF(x,self.h0,self.g),h)))
            self.FixParameters(h,a1,a2)
            self.V=np.asarray(list(map(lambda x: TF(x,self.h0,self.g),h)))
            converged=CheckConvergence(self.V,Vold)
            Vold=self.V
            it=it+1
        #self.CalculateOverlaps()
        return
        
    def TrackDynamics(self,a1,a2):
        return
        
    def FixParameters(self,h,a1,a2):
        maxiter=1000000
        b=0.01
        tolerance=0.01
        fixed=False
        it=0
        while (not fixed) and it<maxiter:
            self.h0=self.h0+b*(pow(np.mean(self.V),2)/np.mean(pow(self.V,2))-a2)
            self.V=np.asarray(list(map(lambda x: TF(x,self.h0,self.g),h)))
            fixed=(abs((pow(np.mean(self.V),2)/np.mean(pow(self.V,2))-a2))/a2 <= tolerance)
        self.g=a1/np.mean(np.asarray(list(map(lambda x: TF(x,self.h0,1),h))))
        if it>=maxiter:
            print("Iter bound reached, fixing failed!")
        return
    
        
    def CalculateOverlaps(self):
        Vmean=np.mean(self.V)
        sp=pow(np.mean(self.V),2)/np.mean(pow(self.V,2))
        for mu in range(self.p):
            for i in range(self.N):
                for j in range(self.N):
                    self.m[mu]=self.m[mu]+(self.V[i]-Vmean)*(self.V[j]-Vmean)*K1cos(self.r[i][mu],self.r[j][mu])
            self.m[mu]=np.sqrt(self.m[mu]/pow(float(self.N),2))/sp
        return
        
#Other Functions
def TF(h,h0,g):
    if (h-h0)>0:
        return g*(h-h0)
    else:
        return 0.0
    
def K1cos(ri,rj):
    thetai=ri*2*math.pi
    thetaj=rj*2*math.pi
    return np.cos(thetai-thetaj)+1


def CheckConvergence(V,Vold):
    tolerance=0.01
    if abs(np.linalg.norm(V-Vold)/np.linalg.norm(V))<tolerance:
        return True
    else:
        return False
    
    