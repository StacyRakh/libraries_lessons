#!/usr/bin/env python
# coding: utf-8

# # Numpy Задание 1

# In[2]:


import numpy as np


# In[3]:


a = np.array ([[1,6], [2,8], [3,11], [3,10],[1,7]])


# In[4]:


a1 = a [0:5,0].copy ()
a2 = a [0:5,1].copy()
mean_a = np.array ([np.mean(a1),np.mean (a2)])
print (mean_a)


# ## Задание 2

# In[5]:


a_centered = np.array ([a1 - mean_a[0], a2 - mean_a[1]])
a_centered = np.array (a_centered.T)
print (a_centered)


# ## Задание 3

# In[8]:


N = len (a)
a_centered_sp = float (a_centered[:5,0]@a_centered [:5,1])
newnum = a_centered_sp/(N-1)
print (a_centered_sp)
print (newnum)


# ## Задание 4
# 

# In[10]:


covar = np.cov (a.T)
newcov = covar [0,1]
print (newcov)


# # Pandas Задание 1

# In[11]:


import pandas as pd


# In[14]:


authors = pd.DataFrame({'author_id': [1,2,3], 'author_name': ['Тургенев', 'Чехов', 'Островский']}, columns = ['author_id', 'author_name'])


# In[16]:


book = pd.DataFrame({'author_id': [1, 1, 1, 2, 2, 3, 3], 
                     'book_title': ['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'],
                    'price': [450, 300, 350, 500, 450, 370, 290]}, columns = ['author_id', 'book_title', 'price'])


# ## Задание 2

# In[20]:


authors_price = pd.merge (authors,book, on = 'author_id', how = 'left')
authors_price


# ## Задание 3

# In[31]:


top_5 = authors_price.nlargest (5,'price')
top_5 = top_5.reset_index (drop = True)
top_5


# ## Задание 4
# 

# In[34]:


authors_min = pd.DataFrame(authors_price.groupby ('author_name')['price'].min())
authors_max = pd.DataFrame(authors_price.groupby ('author_name')['price'].max())
authors_mm = pd.merge (authors_min, authors_max, on = 'author_name', how = 'left')
authors_mean = pd.DataFrame(authors_price.groupby ('author_name')['price'].mean())
authors_stat = pd.merge (authors_mm, authors_mean, on = 'author_name', how = 'left')
authors_stat = authors_stat.rename (columns = {'price_x':'min_price', 'price_y':'max_price', 'price':'mean_price'})
authors_stat 


# ## Задание 5

# In[37]:


cover = ['твердая', 'мягкая', 'мягкая', 'твердая', 'твердая', 'мягкая', 'мягкая']
authors_price ['cover'] = cover
authors_price


# In[39]:


get_ipython().run_line_magic('pinfo', 'pd.pivot_table')


# In[42]:


book_info = pd.pivot_table(authors_price, values=['price'], index=['author_name'],
                       columns = ['cover'], aggfunc=np.sum, fill_value = 0)
book_info


# In[43]:


book_info.to_pickle ('book_info.pkl')


# In[44]:


book_info2 = pd.read_pickle ('book_info.pkl')
book_info2


# In[ ]:




