from numpy import linspace
from scipy.integrate import solve_ivp

#%%
def lorenz(t, xyz):
    x, y, z = xyz
    s, r, b = 10, 28, 8/3. # parameters Lorentz used
    return [s*(y-x), x*(r-z) - y, x*y - b*z]

a, b = 0, 40
t = linspace(a, b, 4000)

sol = solve_ivp(lorenz, [a, b], [1,1,1], t_eval=t)
print(sol.y.shape)