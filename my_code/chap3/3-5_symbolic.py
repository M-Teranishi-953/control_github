import os
import casadi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import patches

x = casadi.SX.sym("x")
y = casadi.SX.sym("y",3,3)

print(casadi.cos(y) - casadi.sin(x))

print(casadi.sin(y)@y) #行列の積
print(casadi.exp(y)*y) #行列の要素積

x = casadi.SX.eye(4)
print(x) #サイズ4の単位行列はなぜかeye
print(x.reshape((2,8))) # 左上から見ていって, 2×8に直す。

x = casadi.SX.sym("x", 3)
y = casadi.DM([1,2,3])
z = x-y
casadi.dot(z,z)

x = casadi.SX.sym("x", 1)
df = casadi.jacobian(x**2, x)
print(df) #x+xになっちゃう
print(casadi.simplify(df)) #シンプルにする方法

df1 = casadi.jacobian(casadi.tan(x), x)
print(df1)
df2 = casadi.jacobian(casadi.sqrt(1+x*x), x)
print(df2)
