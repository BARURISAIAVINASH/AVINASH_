#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1 A1- SLOW TIME SCALE MATRIX
   
   
import numpy as np
from hmmlearn import hmm
np.random.seed(10)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.9, 0.1],[0.1, 0.9]])
                           
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X1, Z1 = model.sample(200)


# In[2]:


X1


# In[ ]:





# In[3]:


Z1


# In[4]:


#2
import numpy as np
from hmmlearn import hmm
np.random.seed(20)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.9, 0.1],[0.1, 0.9]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X2, Z2 = model.sample(200)


# In[5]:


X2


# In[6]:


Z2


# In[7]:


#3
import numpy as np
from hmmlearn import hmm
np.random.seed(30)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.9, 0.1],[0.1, 0.9]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X3, Z3 = model.sample(200)


# In[8]:


X3


# In[9]:


Z3


# In[10]:


#4
import numpy as np
from hmmlearn import hmm
np.random.seed(40)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.9, 0.1],[0.1, 0.9]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X4, Z4= model.sample(200)


# In[11]:


X4


# In[12]:


Z4


# In[13]:


#5
import numpy as np
from hmmlearn import hmm
np.random.seed(50)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.9, 0.1],[0.1, 0.9]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X5, Z5 = model.sample(200)


# In[14]:


X5


# In[15]:


Z5


# In[16]:


#6
import numpy as np
from hmmlearn import hmm
np.random.seed(60)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.9, 0.1],[0.1, 0.9]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X6, Z6 = model.sample(200)


# In[17]:


X6


# In[18]:


Z6


# In[19]:


#7
import numpy as np
from hmmlearn import hmm
np.random.seed(70)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.9, 0.1],[0.1, 0.9]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X7, Z7 = model.sample(200)


# In[20]:


X7


# In[21]:


Z7


# In[22]:


#8
import numpy as np
from hmmlearn import hmm
np.random.seed(80)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.9, 0.1],[0.1, 0.9]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X8, Z8 = model.sample(200)


# In[23]:


X8


# In[24]:


Z8


# In[25]:


#9
import numpy as np
from hmmlearn import hmm
np.random.seed(90)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.9, 0.1],[0.1, 0.9]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X9, Z9 = model.sample(200)


# In[26]:


X9


# In[27]:


Z9


# In[28]:


#10
import numpy as np
from hmmlearn import hmm
np.random.seed(100)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.9, 0.1],[0.1, 0.9]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X10, Z10 = model.sample(200)


# In[29]:


X10


# In[30]:


Z10


# In[31]:


#11
import numpy as np
from hmmlearn import hmm
np.random.seed(200)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.9, 0.1],[0.1, 0.9]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X11, Z11 = model.sample(200)


# In[32]:


X11


# In[33]:


Z11


# In[34]:


#12
import numpy as np
from hmmlearn import hmm
np.random.seed(300)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.9, 0.1],[0.1, 0.9]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X12, Z12 = model.sample(200)


# In[35]:


X12


# In[36]:


Z12


# In[37]:


#13
import numpy as np
from hmmlearn import hmm
np.random.seed(400)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.9, 0.1],[0.1, 0.9]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X13, Z13 = model.sample(200)


# In[38]:


X13


# In[39]:


Z13


# In[40]:


#14
import numpy as np
from hmmlearn import hmm
np.random.seed(500)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.9, 0.1],[0.1, 0.9]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X14, Z14 = model.sample(200)


# In[41]:


X14


# In[42]:


Z14


# In[43]:


#15
import numpy as np
from hmmlearn import hmm
np.random.seed(600)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.9, 0.1],[0.1, 0.9]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X15, Z15 = model.sample(200)


# In[44]:


X15


# In[45]:


Z15


# In[46]:


#16
import numpy as np
from hmmlearn import hmm
np.random.seed(700)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.9, 0.1],[0.1, 0.9]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X16, Z16 = model.sample(200)


# In[47]:


X16


# In[48]:


Z16


# In[49]:


#17
import numpy as np
from hmmlearn import hmm
np.random.seed(800)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.9, 0.1],[0.1, 0.9]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X17, Z17 = model.sample(200)


# In[50]:


X17


# In[51]:


Z17


# In[52]:


#18
import numpy as np
from hmmlearn import hmm
np.random.seed(900)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.9, 0.1],[0.1, 0.9]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X18, Z18 = model.sample(200)


# In[53]:


X18


# In[54]:


Z18


# In[55]:


#19
import numpy as np
from hmmlearn import hmm
np.random.seed(1000)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.9, 0.1],[0.1, 0.9]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X19, Z19 = model.sample(200)


# In[56]:


X19


# In[57]:


Z19


# In[58]:


#20
import numpy as np
from hmmlearn import hmm
np.random.seed(1100)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.9, 0.1],[0.1, 0.9]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X20, Z20 = model.sample(200)


# In[59]:


X20


# In[60]:


Z20


# In[ ]:





# In[ ]:





# In[61]:


#A2 FAST TIME SCALE MATRIX


# In[62]:


#21
import numpy as np
from hmmlearn import hmm
np.random.seed(10)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.2, 0.8],[0.8, 0.2]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X21, Z21 = model.sample(200)


# In[63]:


X21


# In[64]:


Z21


# In[65]:


#22
import numpy as np
from hmmlearn import hmm
np.random.seed(20)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.2, 0.8],[0.8, 0.2]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X22, Z22 = model.sample(200)


# X2

# In[66]:


X22


# In[67]:


Z22


# In[68]:


#23
import numpy as np
from hmmlearn import hmm
np.random.seed(30)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.2, 0.8],[0.8, 0.2]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X23, Z23 = model.sample(200)


# In[69]:


X23


# In[70]:


Z23


# In[71]:


#24
import numpy as np
from hmmlearn import hmm
np.random.seed(40)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.2, 0.8],[0.8, 0.2]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X24, Z24 = model.sample(200)


# In[72]:


X24


# In[73]:


Z24


# In[74]:


#25
import numpy as np
from hmmlearn import hmm
np.random.seed(50)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.2, 0.8],[0.8, 0.2]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X25, Z25 = model.sample(200)


# In[75]:


X25


# In[76]:


Z25


# In[77]:


#26
import numpy as np
from hmmlearn import hmm
np.random.seed(60)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.2, 0.8],[0.8, 0.2]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X26, Z26 = model.sample(200)


# In[78]:


X26


# In[79]:


Z26


# In[80]:


#27
import numpy as np
from hmmlearn import hmm
np.random.seed(70)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.2, 0.8],[0.8, 0.2]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X27, Z27 = model.sample(200)


# In[81]:


X27


# In[82]:


Z27


# In[83]:


#28
import numpy as np
from hmmlearn import hmm
np.random.seed(80)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.2, 0.8],[0.8, 0.2]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X28, Z28 = model.sample(200)


# In[84]:


X28


# In[85]:


Z28


# In[86]:


#29

import numpy as np
from hmmlearn import hmm
np.random.seed(90)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.2, 0.8],[0.8, 0.2]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X29, Z29 = model.sample(200)


# In[87]:


X29


# In[88]:


Z29


# In[89]:


#30
import numpy as np
from hmmlearn import hmm
np.random.seed(100)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.2, 0.8],[0.8, 0.2]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X30, Z30 = model.sample(200)


# In[90]:


X30


# In[91]:


Z30


# In[92]:


#31
import numpy as np
from hmmlearn import hmm
np.random.seed(200)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.2, 0.8],[0.8, 0.2]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X31, Z31 = model.sample(200)


# In[93]:


X31


# In[94]:


Z31


# In[95]:


#32
import numpy as np
from hmmlearn import hmm
np.random.seed(300)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.2, 0.8],[0.8, 0.2]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X32, Z32 = model.sample(200)


# In[96]:


X32


# In[97]:


Z32


# In[98]:


#33
import numpy as np
from hmmlearn import hmm
np.random.seed(400)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.2, 0.8],[0.8, 0.2]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X33, Z33 = model.sample(200)


# In[99]:


X33


# In[100]:


Z33


# In[101]:


#34
import numpy as np
from hmmlearn import hmm
np.random.seed(500)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.2, 0.8],[0.8, 0.2]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X34, Z34 = model.sample(200)


# In[102]:


X34


# In[103]:


Z34


# In[104]:


#35
import numpy as np
from hmmlearn import hmm
np.random.seed(600)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.2, 0.8],[0.8, 0.2]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X35, Z35 = model.sample(200)


# In[105]:


X35


# In[106]:


Z35


# In[107]:


#36
import numpy as np
from hmmlearn import hmm
np.random.seed(700)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.2, 0.8],[0.8, 0.2]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X36, Z36 = model.sample(200)


# In[108]:


X36


# In[109]:


Z36


# In[110]:


#37
import numpy as np
from hmmlearn import hmm
np.random.seed(800)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.2, 0.8],[0.8, 0.2]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X37, Z37 = model.sample(200)


# In[111]:


X37


# In[112]:


Z37


# In[113]:


#38
import numpy as np
from hmmlearn import hmm
np.random.seed(900)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.2, 0.8],[0.8, 0.2]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X38, Z38 = model.sample(200)


# In[114]:


X38


# In[115]:


Z38


# In[116]:


#39
import numpy as np
from hmmlearn import hmm
np.random.seed(1000)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.2, 0.8],[0.8, 0.2]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X39, Z39 = model.sample(200)


# In[117]:


X39


# In[118]:


Z39


# In[119]:


#40
import numpy as np
from hmmlearn import hmm
np.random.seed(1100)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.2, 0.8],[0.8, 0.2]])
                            
model.means_ = np.array([[0.0, 0.0],[3.0,6.0]])
model.covars_ = np.tile(np.identity(2), (2,1,1))
X40, Z40 = model.sample(200)


# In[120]:


X40


# In[121]:


Z40


# In[171]:


A=[X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20]


# In[172]:


A


# In[173]:


import numpy as np




# In[174]:


np.array(A).shape


# In[175]:



B=[X21,X22,X23,X24,X25,X26,X27,X28,X29,X30,X31,X32,X33,X34,X35,X36,X37,X38,X39,X40]


# In[ ]:





# In[176]:


B


# In[177]:


np.array(B).shape
#20 is depth ,200 is rows ,2 columns


# In[178]:


C=[A,B]


# In[ ]:





# In[179]:


C


# In[131]:


np.array(C).shape


# In[132]:


A0=[X1[:,0],X2[:,0],X3[:,0],X4[:,0],X5[:,0],X6[:,0],X7[:,0],X8[:,0],X9[:,0],X10[:,0],X11[:,0],X12[:,0],X13[:,0],X14[:,0],X15[:,0],X16[:,0],X17[:,0],X18[:,0],X19[:,0],X20[:,0]]


# In[133]:


np.array(A0).shape


# In[134]:


A1=[X1[:,1],X2[:,1],X3[:,1],X4[:,1],X5[:,1],X6[:,1],X7[:,1],X8[:,1],X9[:,1],X10[:,1],X11[:,1],X12[:,1],X13[:,1],X14[:,1],X15[:,1],X16[:,1],X17[:,1],X18[:,1],X19[:,1],X20[:,1]]


# In[135]:


np.array(A1).shape


# In[136]:


import matplotlib.pyplot as plt
plt.scatter(A0, A1)


# In[137]:


B0=[X21[:,0],X22[:,0],X23[:,0],X24[:,0],X25[:,0],X26[:,0],X27[:,0],X28[:,0],X29[:,0],X30[:,0],X31[:,0],X32[:,0],X33[:,0],X34[:,0],X35[:,0],X36[:,0],X37[:,0],X38[:,0],X39[:,0],X40[:,0]]


# In[138]:


np.array(B0).shape


# In[139]:


B1=[X21[:,1],X22[:,1],X23[:,1],X24[:,1],X25[:,1],X26[:,1],X27[:,1],X28[:,1],X29[:,1],X30[:,1],X31[:,1],X32[:,1],X33[:,1],X34[:,1],X35[:,1],X36[:,1],X37[:,1],X38[:,1],X39[:,1],X40[:,1]]


# In[140]:


np.array(B1).shape


# In[141]:


plt.scatter(B0, B1,c='red')


# In[142]:


import matplotlib.pyplot as plt

plt.scatter(A0,A1, c='b', marker='x', label='slow_time scale')
plt.scatter(B0, B1, c='r', marker='x', label='fast_time scale')
plt.legend(loc='upper left')
plt.show()


# In[143]:


#


# In[144]:


import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

print(__doc__)

# Number of samples per component
n_samples = 500

# Generate random sample, two components
np.random.seed(0)
C = np.array([[0., -0.1], [1.7, .4]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)


# In[145]:


X


# In[146]:


X.shape


# In[147]:


#


# In[148]:


import numpy as np
from hmmlearn import hmm
np.random.seed(42)

model = hmm.GaussianHMM(n_components=3, covariance_type="full")
model.startprob_ = np.array([0.6, 0.3, 0.1])
model.transmat_ = np.array([[0.7, 0.2, 0.1],
                            [0.3, 0.5, 0.2],
                            [0.3, 0.3, 0.4]])
model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
model.covars_ = np.tile(np.identity(2), (3, 1, 1))
M, Z = model.sample(100)


# In[149]:


import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

print(__doc__)

# Number of samples per component
n_samples = 500

# Generate random sample, two components
#np.random.seed(0)
#C = np.array([[0., -0.1], [1.7, .4]])
#X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          #.7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(M)
        bic.append(gmm.bic(M))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)


# In[150]:


X21.shape


# In[151]:


#######https://bic-berkeley.github.io/psych-214-fall-2016/reshape_and_3d.html


# In[152]:


#######


# In[153]:


A=[A0,A1]


# In[154]:


np.array(A).shape


# In[155]:


import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

print(__doc__)

# Number of samples per component
n_samples = 500

# Generate random sample, two components
#np.random.seed(0)
#C = np.array([[0., -0.1], [1.7, .4]])
#X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          #.7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(X1)
        bic.append(gmm.bic(X1))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)


# In[156]:


import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

print(__doc__)

# Number of samples per component
n_samples = 500

# Generate random sample, two components
#np.random.seed(0)
#C = np.array([[0., -0.1], [1.7, .4]])
#X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          #.7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(X21)
        bic.append(gmm.bic(X21))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)


# In[168]:


A=np.array(A).reshape(-1,2)


# In[169]:


A.shape


# In[170]:


import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

print(__doc__)

# Number of samples per component
n_samples = 500

# Generate random sample, two components
#np.random.seed(0)
#C = np.array([[0., -0.1], [1.7, .4]])
#X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          #.7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(A)
        bic.append(gmm.bic(A))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)


# In[160]:


B=np.array(B).reshape(-1,2)


# In[161]:


np.array(B).shape


# In[162]:


import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

print(__doc__)

# Number of samples per component
n_samples = 500

# Generate random sample, two components
#np.random.seed(0)
#C = np.array([[0., -0.1], [1.7, .4]])
#X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          #.7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(B)
        bic.append(gmm.bic(B))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)


# In[180]:



C=np.array(C).reshape(-1,2)


# In[181]:


np.array(C).shape


# In[182]:


import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

print(__doc__)

# Number of samples per component
n_samples = 500

# Generate random sample, two components
#np.random.seed(0)
#C = np.array([[0., -0.1], [1.7, .4]])
#X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          #.7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(C)
        bic.append(gmm.bic(C))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)


# In[ ]:





# In[ ]:


##########3Observations to volatages


# In[1]:


#####################################################3
import numpy as np
from hmmlearn import hmm
np.random.seed(1050)
import random
model = hmm.MultinomialHMM(n_components=2)
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.9,0.1],
                             [0.1,0.9]]
)
 
model.emissionprob_=np.array([[0.25,0.25,0.25,0.25],
                                [0.25,0.25,0.25,0.2]])




X, y= model.sample(100)

X=X.tolist()
X=np.array(X).reshape(-1,)
Y=[]

for i in range(len(X)):
    if X[i]==0:
        Y.append(random.uniform(2,2.5))
    elif X[i]==1:
        Y.append(random.uniform(2.5,3))
    elif X[i]==2:
        Y.append(random.uniform(3,3.5))
    elif X[i]==3:
        Y.append(random.uniform(3.5,4))
Y


# In[ ]:


############


# In[2]:


import scapy
from scapy.all import sniff
from scapy.all import *
import decimal
from decimal import Decimal


import math
import numpy as np

class MarkovChain():
    def __init__(self, transition_prob, emission_prob):
        
        self.transition_prob = transition_prob
        #print(self.transition_prob)
        self.emission_prob = emission_prob
        #print(self.emission_prob)
        self.states = list(transition_prob.keys())
        #print(self.states)
        self.emitted_states = list(emission_prob.keys())
        
 
    def next_state(self, current_state):
        
        
        
           
        return np.random.choice(
            self.states, 
            p=[self.transition_prob[current_state][next_state] 
               for next_state in self.states]
        )
 
    def next_emitted_state(self, current_state):
       
        return np.random.choice(
            self.emitted_states, 
            p=[self.emission_prob[emitted_state][current_state] 
               for emitted_state in self.emitted_states]
        )

    def generate_states(self, current_state, no=10):
       
        future_states = []
        emitted_states=[]
        x=[]
        for i in range(no):
            next_state = self.next_state(current_state)
            #print("Next state is",next_state)
            emitted_states.append(self.next_emitted_state(next_state))
            future_states.append(next_state)
            current_state = next_state
        x=emitted_states
        return [x[-1], current_state]

transition_prob = {'H': {'H': 0.3, 'NH': 0.7}, 'NH': {'H': 0.1, 'NH': 0.9}}

B = {'1': {'H': 0.25, 'NH': 0.25},
     '2': {'H': 0.25, 'NH': 0.25},
     '3': {'H': 0.25, 'NH': 0.25},
     '4': {'H': 0.25, 'NH': 0.25},
     
     }



chain = list(np.zeros(100))

for i in range(5):
    chain[i] = MarkovChain(transition_prob=transition_prob,emission_prob=B)
    [All_Samples, cur_state]=chain[i].generate_states(current_state='H', no=2)
    #print(All_Samples)
    #print(cur_state)
    

   

   
    X=[]
    for index in range(100):
        
        [All_Samples, cur_state]=chain[i].generate_states(current_state=cur_state, no=1)
     
        X.append(All_Samples)
    
for i in range(0, len(X)): 
    X[i] = int(X[i])       
Y=[]

for i in range(len(X)):
    if X[i]==1:
        Y.append(random.uniform(2,2.5))
    elif X[i]==2:
        Y.append(random.uniform(2.5,3))
    elif X[i]==3:
        Y.append(random.uniform(3,3.5))
    elif X[i]==4:
        Y.append(random.uniform(3.5,4))
print(Y)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




