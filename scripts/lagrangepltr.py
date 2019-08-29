import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt



def ode(y, x, lam, n0, delta, y0):
    c = 3e8 / (n0-delta*np.exp(y/y0))
    dydx = ((1/(lam * c))**2 - 1)
    return dydx

vertex = -2000

#use southpole_2015 model data
n0 = 1.78 
delta = 0.432
y0 = 77.0

x = np.flip(np.linspace(0,5000))

dist = np.array([0.0 for j in range(20)])
angs = np.array([0.0 for j in range(20)])

lam = 0.6e-8
yinit = odeint(ode, vertex, x, args=(lam, n0, delta, y0))
ypinit = ode(vertex, x[0], lam, n0, delta, y0)
ylaunchvect = np.array([1, ypinit])/np.linalg.norm([1, ypinit])
for i in range(1, 21):
    y1 = odeint(ode, vertex, x, args=(lam, n0, delta, y0))
    dist[i-1] = y1[-1] - yinit[-1]
    yp1 = ode(vertex, x[0], lam, n0, delta, y0)
    y1launchvect = np.array([1, yp1])/np.linalg.norm([1, yp1])
    angs[i-1] = np.arccos(np.dot(ylaunchvect, y1launchvect))
    plt.plot(x, y1, label="lam=" + str(lam))
    lam += 0.1e-8

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('radial distance [m]')
plt.ylabel('z axis [m]')
plt.tight_layout()
#plt.savefig('rays.pdf')
plt.savefig('rays.pdf', bbox_inches='tight')
#plt.tight_layout()
plt.show()
plt.clf()

plt.plot(dist, angs, 'g^')
plt.xlabel('distance (on z-axis) [m]')
plt.ylabel('angles [rad]')
plt.title(r'$\theta (\delta)$ w.r.t. initial ray')
plt.savefig('angles_vs_dist.pdf')
plt.show()
plt.clf()

lambdas = np.array([0.0 for j in range(20)])
for j in range(20):
    lambdas[j] = 0.6e-8 + j*0.1e-8

x2 = np.linspace(0.6e-8, 2.1e-8)

plt.plot(lambdas, dist, 'g.')

def fn(n0, delta, mu, y0, xi, xf, lam):
    c0 = 3e8
    rho = (c0*lam)**2
    umu = n0 - delta*np.exp(mu/y0)
    kappa = np.exp(2*delta * (xi-xf)/y0)
    return n0 - np.sqrt((rho * umu**2)/(umu**2 + kappa*rho - kappa*(umu**2)))

plt.plot(x2, y0*np.log(fn(n0, delta, vertex, y0, 0, 1000, x2)/fn(n0, delta, vertex, y0, 0, 1000, 0.6e-8)))
plt.show()
