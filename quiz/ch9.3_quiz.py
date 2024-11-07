import sympy as sym

t = sym.Symbol('t')
T = sym.Symbol('T')
a0, a1, a2, a3, a4, a5 = sym.symbols('a0 a1 a2 a3 a4 a5')

s = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
display(s)