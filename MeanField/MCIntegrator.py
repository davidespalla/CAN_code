import numpy as np

def MCIntegrate2D(f,x1min,x1max,x2min,x2max,*argv):
    argv=np.asarray(argv)
    npoints=100000
    I=0
    V=abs(x1max-x1min)*abs(x2max-x2min)
    x1=np.random.uniform(x1min,x1max,npoints)
    x2=np.random.uniform(x2min,x2max,npoints)
    X=np.c_[x1.T,x2.T]
    I=np.mean(list(map(lambda X:f(X,argv),X)))*V
    return I

def MCIntegrate3D(f,x1min,x1max,x2min,x2max,x3min,x3max,*argv):
    argv=np.asarray(argv)
    npoints=100000
    I=0
    V=abs(x1max-x1min)*abs(x2max-x2min)*abs(x3max-x3min)
    x1=np.random.uniform(x1min,x1max,npoints)
    x2=np.random.uniform(x2min,x2max,npoints)
    x3=np.random.uniform(x3min,x3max,npoints)
    X=np.c_[x1.T,x2.T,x3.T]
    I=np.mean(list(map(lambda X:f(X,argv),X)))*V
    return I

def MCIntegrate4D(f,x1min,x1max,x2min,x2max,x3min,x3max,x4min,x4max,*argv):
    argv=np.asarray(argv)
    npoints=100000
    I=0
    V=abs(x1max-x1min)*abs(x2max-x2min)*abs(x3max-x3min)*abs(x4max-x4min)
    x1=np.random.uniform(x1min,x1max,npoints)
    x2=np.random.uniform(x2min,x2max,npoints)
    x3=np.random.uniform(x3min,x3max,npoints)
    x4=np.random.uniform(x4min,x4max,npoints)
    X=np.c_[x1.T,x2.T,x3.T,x4.T]
    I=np.mean(list(map(lambda X:f(X,argv),X)))*V
    return I

#REQIRES f to be in the form f([variables to integrate],[arguments])