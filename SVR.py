import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# tools untuk melakukan feature scalling
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values # memulai dari indeks 1 dan tidak menggunakan indeks terakhir
y = dataset.iloc[:,-1].values

print(x)
print()
print(y) # karena y masih dalam bentuk array 1D kita perlu mengubahnya ke bentuk array 2D karena kita perlu membuat bentuknya menjadi array 2D untuk melakuykan feature scalling
print()

#cara merubah y ke bentuk 2D dengan baris sebanyak elemen y dan 1 kolom
y = y.reshape(len(y), 1) # merubah bentuk y menggunakan function reshape(*Baris ,*Kolom)
print(y)
print()

# Feature Scalling
sc_x = StandardScaler()
x = sc_x.fit_transform(x)

# CREATE NEW STANDARD SCALLER OBJECT
# That's because you know when you fit your object as sc on your data, while it is going to compute the mean and the standard deviation of that same variable
# and therefore, since of course we don't have the same mean and same standard deviation for our levels here and our salaries ,
# well obviously we have to create two standard scaler object one that will be fitted to X in order to compute the mean and standard deviation of the position levels,
# and one that will be fitted to Y to indeed compute the mean and the standard deviation of the salaries.
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

print(x)
print()
print(y)
print()