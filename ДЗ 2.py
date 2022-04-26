#!/usr/bin/env python
# coding: utf-8

# # Домашнее задание 2

# ## Задание 1

# In[10]:


import numpy as np
import pandas as pd


# In[11]:


import matplotlib.pyplot as plt


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


x = list (i for i in range (1,8))
y = [3.5, 3.8, 4.2, 4.5, 5, 5.5, 7]


# In[6]:


plt.plot (x,y)
plt.show ()


# In[7]:


plt.scatter (x,y)
plt.show ()


# # Задание 2

# In[8]:


t = np.linspace (0,10,51)


# In[9]:


f = np.cos (t)


# In[10]:


plt.axis ([0.5, 9.5, -2.5, 2.5])
plt.plot (t,f, color = 'green')
plt.title ('График f(t)')
plt.xlabel ('Значения t')
plt.xlabel ('Значения f')
plt.show ()



# ## Задание 3

# In[22]:


x = np.linspace (-3,3,51)


# In[23]:


y1 = x**2
y2 = 2*x + 0.5
y3 = -3*x - 1.5
y4 = np.sin (x)


# In[24]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# In[25]:


fig, ax = plt.subplots (nrows = 2, ncols = 2)
ax1, ax2, ax3, ax4 = ax.flatten ()
ax1.plot (x,y1)
ax1.set_title ('График y1')
ax1.set_xlim ([-5,5])
ax2.plot (x,y2)
ax2.set_title ('График y2')
ax3.plot (x,y3)
ax3.set_title ('График y3')
ax4.plot (x,y4)
ax4.set_title ('График y4')
fig.set_size_inches (8,6)
plt.subplots_adjust (wspace = 0.3, hspace = 0.3)


# ## Задание 4

# In[7]:


import numpy as np
import pandas as pd


from scipy.stats import mode
import warnings
warnings.filterwarnings ('ignore')


# In[8]:


hwdata = 'C:/Users/Анастасия Рахманина/Desktop/creditcard.csv'


# In[11]:


df = pd.read_csv (hwdata, sep = ',')


# In[12]:


df.isna().sum()


# In[17]:


special_data = df['Class'].value_counts()
special_data


# In[19]:


import matplotlib.pyplot as plt
special_data.plot(kind = 'bar')
plt.show()


# In[20]:


special_data.plot(kind = 'bar', logy = True)
plt.show()


# С остальной частью задания надо разбираться дальше, пока возникли трудности

# # Задачи на повторение

# ## 1

# In[7]:


import numpy as np
import pandas as pd


# In[24]:


a = np.arange (12,24)
print (a)


# ## 2

# In[11]:


a.reshape (2,6)


# In[12]:


a.reshape (3,4)


# In[13]:


a.reshape (4,3)


# In[14]:


a.reshape (6,2)


# In[15]:


a.reshape (12,1)


# ## 3

# In[16]:


a.reshape (-1,6)


# In[17]:


a.reshape (-1, 4)


# In[18]:


a.reshape (12,-1)


# In[19]:


a.reshape (3,-1)


# In[20]:


a.reshape (4,-1)


# ## 4

# Вероятно, нет, потому что в данном случае, каждая строка является отдельным массивом (списком). Задать одним списком столбец не выходит.

# ## 5

# In[27]:


b = np.random.randn (3,4)
b1 = b.flatten ()
print (b1)


# ## 6

# In[32]:


a = np.arange (20,0,-2)
print (a)


# ## 7

# In[36]:


b = np.arange (20,1,-2)
b.reshape (1,10)


# В данном случае, мы задаем двумерный массив, в котором имеется только 1 строка. 

# ## 8

# In[45]:


a = np.zeros ((2,2))
b = np.ones ((3,2))
v = np.concatenate ((a,b), axis = 0)
v.size 


# ## 9

# In[114]:


a = np.arange (0,12)
A = a.reshape (4,3)
At = A.T
B = A.dot (At)
B


# In[89]:


print (np.linalg.matrix_rank (B))


# In[90]:


print (np.linalg.det (B))


# Определитель равен нулю. Нельзя вычислить обратную матрицу. 

# ## 10

# In[91]:


import random
random.seed (42)


# ## 11

# In[92]:


c = np.random.randint (0,16, (1, 16))
print (c)


# ## 12

# In[93]:


c= c.reshape (4,4)
c


# In[94]:


D = B + c*10
D


# In[95]:


print (np.linalg.det (D))


# In[96]:


print (np.linalg.matrix_rank (D))


# In[97]:


D_inv = np.linalg.inv (D)
print (D_inv)


# ## 13

# In[98]:


zeroes = np.where (D_inv<0)
ones = np.where (D_inv>0)


# In[99]:


D_inv[zeroes] = 0
D_inv[ones]=1
print (D_inv)


# In[128]:


reasc = np.where (D_inv == 0)
reasB = np.where (D_inv == 1)
D_inv [zeroes] = c[zeroes] 
D_inv [ones] = B [ones]
E = D_inv
E


# In[ ]:





# In[ ]:




