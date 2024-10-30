# Finite difference solver for the 2D wave equation with absorptive boundary conditions.
# Based on Hans Petter Langtangen online book "Finite Difference Computing with PDEs"
# and accompanying code https://github.com/hplgit/fdm-book.git

"""
2D wave equation solved by finite differences::

  u_log, dt = solver(I, V, f, c, Lx, Ly, Nx, Ny, dt, T)

Solve the 2D wave equation u_tt = c^2 (u_xx + u_yy) + f(x,t) on (0,L) with
u_x=-u_t/c, u_x=u_t/c, u_y=-u_t/c, u_y=u_t/c on the boundaries and initial 
conditions u(x,y,0)=I and u_t(x,y,0)=V.

Nx and Ny are the total number of mesh cells in the x and y
directions. The mesh points are numbered as (0,0), (1,0), (2,0),
..., (Nx,0), (0,1), (1,1), ..., (Nx, Ny).

dt is the time step. If dt<=0, an optimal time step is used.
T is the stop time for the simulation.

I, V, f are functions: I(x,y), V(x,y), f(x,y,t). V and f
can be specified as None or 0, resulting in V=0 and f=0.
"""
from tqdm import tqdm
import numpy as np

def solver(I, V, f, c, Lx, Ly, Nx, Ny, dt, T):

    x = np.linspace(0, Lx, Nx)  # Mesh points in x dir
    y = np.linspace(0, Ly, Ny)  # Mesh points in y dir
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    x = x.reshape(-1,1)
    y = y.reshape(1,-1)  

    stability_limit = (1/float(c))*(1/np.sqrt(1/dx**2 + 1/dy**2))
    if dt <= 0:                # max time step?
        safety_factor = -dt    # use negative dt as safety factor
        dt = safety_factor*stability_limit
    elif dt > stability_limit:
        print('error: dt=%g exceeds the stability limit %g' % \
              (dt, stability_limit))
    Nt = int(np.ceil(T/float(dt)))+1
    t = np.linspace(0, Nt*dt-dt, Nt)    # mesh points in time
    Cx2 = (c*dt/dx)**2;  Cy2 = (c*dt/dy)**2    # help variables
    dt2 = dt**2

    # Allow f and V to be None or 0
    if f is None or f == 0:
        f = (lambda x, y, t: np.zeros((x.shape[0], y.shape[1])))
        # or simpler: x*y*0
    if V is None or V == 0:
        V = (lambda x, y: np.zeros((x.shape[0], y.shape[1])))

    u     = np.zeros((Nx,Ny))   # Solution array
    u_n   = np.zeros((Nx,Ny))   # Solution at t-dt
    u_nm1 = np.zeros((Nx,Ny))   # Solution at t-2*dt
    u_log = np.zeros((Nx,Ny,Nt))

    It = range(0, t.shape[0]+1)

    # Load initial condition into u_n
    u_n[:,:] = I(x, y)

    # Special formula for first time step
    n = 0
    f_a = f(x, y, t[n])
    V_a = V(x, y)
    u = advance(
        u, u_n, u_nm1, f_a, Cx2, Cy2, dt2, V=V_a, step1=True)
    u_log[:,:,0] = u

    # Update data structures for next step
    u_nm1, u_n, u = u_n, u, u_nm1

    for n in tqdm(It[1:-1]):
        f_a = f(x, y, t[n])  
        u = advance(u, u_n, u_nm1, f_a, Cx2, Cy2, dt2)

        # Update data structures for next step
        u_log[:,:,n] = u
        u_nm1, u_n, u = u_n, u, u_nm1
        
    # t = t[0:-1]
    return u_log, x, y, t, dt

def advance(u, u_n, u_nm1, f_a, Cx2, Cy2, dt2, V=None, step1=False):
    if step1:
        dt = np.sqrt(dt2) 
        Cx2 = 0.5*Cx2;  Cy2 = 0.5*Cy2; dt2 = 0.5*dt2
        D1 = 1;  D2 = 0
    else:
        D1 = 2;  D2 = 1
    u_xx = u_n[:-2,1:-1] - 2*u_n[1:-1,1:-1] + u_n[2:,1:-1]
    u_yy = u_n[1:-1,:-2] - 2*u_n[1:-1,1:-1] + u_n[1:-1,2:]
    u[1:-1,1:-1] = D1*u_n[1:-1,1:-1] - D2*u_nm1[1:-1,1:-1] + \
                   Cx2*u_xx + Cy2*u_yy + dt2*f_a[1:-1,1:-1]
    if step1:
        u[1:-1,1:-1] += dt*V[1:-1, 1:-1]
    
    # Boundary condition
    kappa = np.sqrt(Cy2)
    beta = (kappa-1)/(kappa+1)
    j = 0
    u[:,j] = u_n[:,j+1] + beta*(u[:,j+1] - u_n[:,j])
    j = u.shape[1]-1
    u[:,j] = u_n[:,j-1] + beta*(u[:,j-1] - u_n[:,j])

    kappa = np.sqrt(Cx2)
    beta = (kappa-1)/(kappa+1)
    i = 0
    u[i,:] = u_n[i+1,:] + beta*(u[i+1,:] - u_n[i,:])
    i = u.shape[0]-1
    u[i,:] = u_n[i-1,:] + beta*(u[i-1,:] - u_n[i,:])

    return u