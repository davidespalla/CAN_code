import numpy as np

class Network:
    def __init__(self,N,p):
        self.N=N
        self.p=p
        self.V=np.zeros(N)
        self.J=np.zeros((N,N))
        self.m=np.zeros(p)
            
    def SetJ(self):
        
    def SetV0(self,a1):
        self.V=np.random.uniform(0,1,self.N)
        self.V=self.a1/mean(self.V)
        return 
        
    def EvolveDynamics(self,a1,a2):
        h=np.zeros(self.N)
        converged=False
        g=
        th=
        while (not converged):
            h=np.dot(self.J,self.V)
            self.V=asarray(list(map(lambda h: TF(h,h0,g),h)))
            th,g=FixParameters()
            self.V=asarray(list(map(lambda h: TF(h,h0,g),h)))
            converged=CheckConvergence()
        CalculateOverlaps()
        
    def TrackDynamics(self,a1,a2):
        