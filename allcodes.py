# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


a = [1,0,1 ,1 ,0 ,1 ,1 ,1]
b = [0 ,0 ,0 ,1 ,1 ,1 ,0 ,1]
plt.plot(a,b ,'ro')


# In[ ]:


import numpy as np 
import matplotlib.pyplot as plt 
X = np.linspace(-np.pi, np.pi, 256)
C,S = np.cos(X), np.sin(X)
plt.plot(X, C)
plt.plot(X, S)
plt.show()


# In[ ]:


plt.plot(X, C, color="red", linewidth=10.0, linestyle="-")
plt.plot(X, S, color="green", linewidth=5.0, linestyle="-")
plt.plot(X, C)
plt.plot(X, S)
plt.show()


# In[ ]:


plt.plot(X, C, color="red", linewidth=10.0, linestyle="-")
plt.plot(X, S, color="green", linewidth=5.0, linestyle="-")
plt.xlim(-4.0, 4.0)
plt.ylim(-1.0, 1.0)
plt.plot(X, C)
plt.plot(X, S)
plt.show()


# In[ ]:


plt.plot(X, C, color="red", linewidth=10.0, linestyle="-")
plt.plot(X, S, color="green", linewidth=5.0, linestyle="-")
plt.xlim(-4.0, 4.0)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
plt.yticks([-1,0,+1])
plt.ylim(-1.0, 1.0)
plt.plot(X, C)
plt.plot(X, S)
plt.show()


# In[ ]:


plt.plot(X, C, color="red", linewidth=10.0, linestyle="-")
plt.plot(X, S, color="green", linewidth=5.0, linestyle="-")
plt.xlim(-4.0, 4.0)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],[r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
plt.yticks([-1,0,+1])
plt.yticks([-1,0,+1],[r'$-a$', r'$0$', r'$+a$'])
plt.ylim(-1.0, 1.0)
plt.plot(X, C)
plt.plot(X, S)
plt.show()


# In[ ]:


plt.plot(X, C, color="red", linewidth=10.0, linestyle="-")
plt.plot(X, S, color="green", linewidth=5.0, linestyle=":")
plt.xlim(-4.0, 4.0)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],[r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
plt.yticks([-1,0,+1])
plt.yticks([-1,0,+1],[r'$-a$', r'$0$', r'$+a$'])
plt.ylim(-1.0, 1.0)
plt.plot(X, C)
plt.plot(X, S)
plt.show()


# In[ ]:


plt.plot(X, C, color="red", linewidth=10.0, linestyle="-")
plt.plot(X, S, color="green", linewidth=5.0, linestyle=":")

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

plt.xlim(-4.0, 4.0)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],[r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
plt.yticks([-1,0,+1])
plt.yticks([-1,0,+1],[r'$-a$', r'$0$', r'$+a$'])
plt.ylim(-1.0, 1.0)
plt.plot(X, C)
plt.plot(X, S)
plt.show()


# ### Simple Plot - Sine and Cosine

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
X = np.linspace(-np.pi, np.pi, 256)
C, S = np.cos(X), np.sin(X)
plt.plot(X, C)
plt.plot(X, S)
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6), dpi=80)

X = np.linspace(-np.pi, np.pi, 256)
C, S = np.cos(X), np.sin(X)
plt.plot(X, C)
plt.plot(X, S)
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-np.pi, np.pi, 256)
C, S = np.cos(X), np.sin(X)
plt.plot(X, C, color="red", linewidth=10.0, linestyle="-")
plt.plot(X, S, color="green", linewidth=5.0, linestyle="-")
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-np.pi, np.pi, 256)
C, S = np.cos(X), np.sin(X)
plt.plot(X, C, color="red", linewidth=10.0, linestyle="-")
plt.plot(X, S, color="green", linewidth=5.0, linestyle="-")
plt.xlim(-4.0, 4.0)
plt.ylim(-1.0, 1.0)
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-np.pi, np.pi, 256)
C, S = np.cos(X), np.sin(X)
plt.plot(X, C, color="red", linewidth=10.0, linestyle="-")
plt.plot(X, S, color="green", linewidth=5.0, linestyle="-")
plt.xlim(-4.0, 4.0)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
plt.yticks ([-1, 0, 1])
plt.ylim(-1.0, 1.0)
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-np.pi, np.pi, 256)
C, S = np.cos(X), np.sin(X)
plt.plot(X, C, color="red", linewidth=10.0, linestyle="--")
plt.plot(X, S, color="green", linewidth=5.0, linestyle=":")
plt.xlim(-4.0, 4.0)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

plt.yticks ([-1, 0, 1])
plt.yticks ([-1, 0, 1], [r'$-a$', r'$0$', r'$+a$'])
plt.ylim(-1.0, 1.0)
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-np.pi, np.pi, 256)
C, S = np.cos(X), np.sin(X)
plt.plot(X, C, color="red", linewidth=10.0, linestyle="--")
plt.plot(X, S, color="green", linewidth=5.0, linestyle=":")

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

plt.xlim(-4.0, 4.0)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

plt.yticks ([-1, 0, 1])
plt.yticks ([-1, 0, 1], [r'$-a$', r'$0$', r'$+a$'])
plt.ylim(-1.0, 1.0)
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-np.pi, np.pi, 256)
C, S = np.cos(X), np.sin(X)
plt.plot(X, C, color="red", linewidth=10.0, linestyle="--", label="cosine")
plt.plot(X, S, color="green", linewidth=5.0, linestyle=":", label="sine")

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

plt.xlim(-4.0, 4.0)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

plt.yticks ([-1, 0, 1])
plt.yticks ([-1, 0, 1], [r'$-a$', r'$0$', r'$+a$'])
plt.ylim(-1.0, 1.0)

t = 2*np.pi/3
plt.plot([t, t], [0, np.cos(t)], color='blue', linewidth=2.5, linestyle="-.")
plt.scatter([t, ], [np.cos(t), ], 50, color='blue')
plt.annotate(r'$sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',
            xy=(t, np.sin(t)), xycoords='data',
            xytext=(+10, +30), textcoords='offset points', fontsize= 16,
            arrowprops = dict(arrowstyle="->", connectionstyle="arc3, rad=.2"))

plt.plot([t, t], [0, np.sin(t)],
        color='pink', linewidth=2, linestyle="--")
plt.plot([t, t], [0, np.sin(t)],
        color='blue', linewidth=2.5, linestyle="-.")
plt.scatter([t, ], [np.cos(t), ], 50, color='blue')
plt.annotate(r'$sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',
            xy = (t, np.sin(t)), xycoords='data', xytext=(-90, -50), textcoords='offset points',
            fontsize=16,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.2"))
plt.plot(X, C)
plt.plot(X, S)
plt.legend(loc='upper left')
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
X = np.linspace(-np.pi, np.pi, 200)
C, S = np.cos(X), np.sin(X)
plt.subplot(2,2,1)
plt.plot(X,C,color='red',linewidth=2)
plt.subplot(2,2,2)
plt.plot(X, S)
plt.subplot(223)
plt.plot(X, X*X)
plt.subplot(224)
plt.plot(X, X) 


# ### Subplots

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
X = np.linspace(-np.pi, np.pi, 100)
S = np.cos(X)
C = np.sin(X)
SC=S*C
plt.subplot(221)
plt.plot(X, S)
plt.title('sin(x)')
plt.subplot(222)
plt.plot(X, C)
plt.title('cos(x)')
plt.subplot(212)
plt.plot(X, SC)
plt.title('sin(x)*cos(x)') 


# ### Subplot example in which one plot is spanned horizontally

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
X = np.linspace(-np.pi, np.pi, 100)
S = np.cos(X)
C = np.sin(X)
SC=S*C
plt.subplot(221)
plt.plot(X, S)
plt.title('sin(x)')
plt.subplot(222)
plt.plot(X, C)
plt.title('cos(x)')
plt.subplot(212)
plt.plot(X, SC)
plt.title('sin(x)*cos(x)') 


# ### Subplot example in which one plot is spanned Vertically 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
X = np.linspace(-np.pi, np.pi, 100)
S = np.cos(X)
C = np.sin(X)
SC=S*C
plt.subplot(222)
plt.plot(X, S)
plt.title('sin(x)')
plt.subplot(224)
plt.plot(X, C)
plt.title('cos(x)')
plt.subplot(121)
plt.plot(X, SC)
plt.title('sin(x)*cos(x)') 


# ### Python programming illustrating the scatter() plot function 

# In[33]:


import numpy as np
import matplotlib.pyplot as plt
n=1024
X=np.random.normal(0,1,n)
Y=np.random.rand(n)
plt.figure()
plt.scatter(X,Y,color='green',alpha=0.5) 


# In[34]:


import numpy as np
import matplotlib.pyplot as plt
n=10
X=np.arange(n)
Y1=(1-X/float(n))*np.random.uniform(0.5,1.0,n)
Y2=(1-X/float(n))*np.random.uniform(0.5,1.0,n)
plt.figure()
plt.bar(X,+Y1,facecolor='#9999ff',edgecolor='white')
plt.bar(X,-Y2,facecolor='#ff9999',edgecolor='white')
for x,y in zip(X,Y1):
    plt.text(x,y+0.02,'%.2f'%y,ha='center',va='bottom')
for x,y in zip(X,Y2):
    plt.text(x,-y-0.15,'%.2f'%y,ha='center',va='bottom')
plt.ylim(-1.1,1.1) 


# 

# ### Scipy: plotting 

# In[35]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as spip
from mpl_toolkits.mplot3d import Axes3D

x = np.arange(-5.01, 5.01, 0.25)
y = np.arange(-5.01, 5.01, 0.25)
xx, yy = np.meshgrid(x, y)
z = np.sin(xx**2+yy**2)
f = spip.interp2d(x, y, z, kind='cubic')
xnew = np.arange(-5.01, 5.01, 1e-2)
ynew = np.arange(-5.01, 5.01, 1e-2)
plt.figure()

znew = f(xnew, ynew)
plt.plot(x, z[0, :], 'ro-', xnew, znew[0, :], 'b-')
plt.show()

xx1, yy1 = np.meshgrid(xnew, ynew)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter3D(xx,yy,z)
plt.show()
fig1 = plt.figure()
ax = Axes3D(fig1)
ax.plot_surface(xx1,yy1,f(xnew,ynew))
plt.show()


# ### Program for fitting the curve to a data with the constraints on the parameters

# In[37]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# Fitting function defintion
def func(x, a, b, c):
    return a * np.exp(-b * x) + c
# Generate some data with noise
xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
np.random.seed(1729)
y_noise = 0.2 *np.random.normal(size=xdata.size)
ydata = y + y_noise 
plt.plot(xdata, ydata, 'b-', label='data')
# Fit the data and obtain the parameters
popt, pcov = curve_fit(func, xdata, ydata)
# Display the parameter values
popt
# Plot the data and fitted curve
plt.plot(xdata, func(xdata, *popt), 'r-',
label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
# Constarined optimization
# If we require the parameters to lie in a given region
# i.e., 0 <= a <= 3; 0 <= b <= 1; 0 <= c <= 0.5 
popt, pcov = curve_fit(func, xdata, ydata,
bounds = ([0, 0, 0], [3., 1.,0.5]))
popt
plt.plot(xdata, func(xdata, *popt), 'g--',
label='fit: a=%5.3f, b=%5.3f, c=%5.3f' %
tuple(popt))
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# ### Python program for finding the minimum of a scalar function 

# In[38]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as spop
def f(x):
    return x**2+10*np.sin(x)
plt.figure()
x = np.arange(-10,10,0.1)
plt.plot(x,f(x))
plt.show()
result = spop.minimize(f, x0=0)
print(result)
print('Minimum is at: %.4f' %result.x)
plt.plot(result.x,f(result.x),'o') 


# ### Q2

# In[39]:


import numpy as np
# Define the function that we are interested in
def sixhump(x):
    return ((4 - 2.1*x[0]**2 + x[0]**4 / 3.) * x[0]**2 + x[0] * x[1] + (-4 + 4*x[1]**2) * x[1] **2)
# Make a grid to evaluate the function (for plotting)
x = np.linspace(-2, 2)
y = np.linspace(-1, 1)
xg, yg = np.meshgrid(x, y)

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(sixhump([xg, yg]), extent=[-2, 2, -1, 1])
plt.colorbar()
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xg, yg, sixhump([xg, yg]),
cmap=plt.cm.jet, linewidth=0, antialiased=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('Six-hump Camelback function')
from scipy import optimize
x_min = optimize.minimize(sixhump, x0=[0, 0])
plt.figure()

# Show the function in 2D
plt.imshow(sixhump([xg, yg]), extent=[-2, 2, -1, 1])
plt.colorbar()
# And the minimum that we've found:
plt.scatter(x_min.x[0], x_min.x[1])
plt.show()


# ### Histogram Comparisons

# In[40]:


import numpy as np
from matplotlib import pyplot as plt
# Generates 2 sets of observations
samples1 = np.random.normal(0, size=1000)
samples2 = np.random.normal(1, size=1000)
# Compute a histogram of the sample
bins = np.linspace(-4, 4, 30)
histogram1, bins = np.histogram(samples1, bins=bins, normed=True)
histogram2, bins = np.histogram(samples2, bins=bins, normed=True)
plt.figure(figsize=(6, 4))
plt.hist(samples1, bins=bins, normed=True, label="Samples 1")
plt.hist(samples2, bins=bins, normed=True, label="Samples 2")
plt.legend(loc='best')
plt.show()


# ### Python program for plotting histograms and probability density functions using SciPy 

# In[43]:


import numpy as np
samples = np.random.normal(size=1000)
bins = np.arange(-4, 5)
print(bins)
histogram = np.histogram(samples, bins=bins, normed=True)[0]
bins = 0.5*(bins[1:] + bins[:-1])
print(bins)
from scipy import stats
pdf = stats.norm.pdf(bins) # norm is a distribution object
plt.plot(bins, histogram)
plt.plot(bins, pdf)

