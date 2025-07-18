import os
import casadi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import patches

a = 10

R = casadi.SX.sym('R')
x_list = [casadi.SX.sym(f'x{i}') for i in range(5)]
y_list = [casadi.SX.sym(f'y{i}') for i in range(5)]
r_list = [R,R,R,2*R,2*R]

x_nlp = casadi.vertcat(R, *x_list, *y_list)
f_nlp = -R
g_nlp = []

g_nlp.append(R)

for i in range(5):
    g_nlp.append(x_list[i] - r_list[i])
    g_nlp.append(a-x_list[i] - r_list[i])
    g_nlp.append(y_list[i] - r_list[i])
    g_nlp.append(a-y_list[i] - r_list[i])

for j in range(5):
    for i in range(0, j):
        eq = (x_list[i] - x_list[j])**2 + (y_list[i]-y_list[j])**2 \
                - (r_list[i] + r_list[j])**2
        g_nlp.append(eq)

g_nlp = casadi.vertcat(*g_nlp)

nlp = {'x': x_nlp,
        'f': f_nlp,
        'g': g_nlp
        }

S = casadi.nlpsol('S', 'ipopt', nlp)
print(S)

r = S(x0=[1,1,3,8,7,3,1,2,8,3,7],\
      lbg=[0]*31, ubg=[np.inf]*31)
x_opt = np.array(r['x']).ravel()
print('x_opt: ', x_opt)

R_opt = x_opt[0]
coord_opt = x_opt[1:].reshape(2,5).T

fig, ax = plt.subplots(figsize=(6,6))

for i in range(5):
       radius = R_opt if i <= 2 else 2*R_opt
       circle = patches.Circle(xy=coord_opt[i],radius=radius,fill=False)
       ax.add_patch(circle)
       ax.scatter(*coord_opt[i],marker="x")

rect = patches.Rectangle([0,0],10,10,fill=False)
ax.add_patch(rect)

ax.set_xlim(0,10)
ax.set_ylim(0,10)
ax.axis("equal")

plt.savefig("images/chap3_NLP_packing_1.png")
plt.show()
