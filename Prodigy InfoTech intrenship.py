#!/usr/bin/env python
# coding: utf-8

# # Task 1

# # Implement a linear regression model to predict the prices of house based on their square footage and the number of bedrooms and bathrooms.

# In[73]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[74]:


# Sample data (square footage, number of bedrooms, number of bathrooms, price)
data = np.array([
    [2000, 2, 1, 600000],
    [2500, 3, 2, 800000],
    [4000, 4, 2, 1000000],
    [5000, 4, 3, 1200000],
    [6000, 5, 3, 1400000]
])


# In[75]:


# Split the data into features (X) and target variable (y)
X = data[:, :-1]  # Features: square footage, number of bedrooms, number of bathrooms
y = data[:, -1]   # Target variable: price


# In[76]:


# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[77]:


# Make predictions
y_pred = model.predict(X_test)


# In[78]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[79]:


# Example prediction
example_house = np.array([[4000, 3, 2]])  # Square footage: 4000, Bedrooms: 3, Bathrooms: 2
predicted_price = model.predict(example_house)
print("Predicted price for the example house:", predicted_price)


# In[ ]:




