import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV generated from C++
data = pd.read_csv("pi_comparison.csv")

# Print the data
print(data)
print((data.iloc[::2,1 ]))


plt.figure(figsize=(15, 5))
plt.plot(data.iloc[::2, 1],data.iloc[::2, 2],  '*-', label='MonteCarlo')
plt.axhline(y=3.141592653589793, color='r', linestyle='--', label='Actual π')
plt.plot(data.iloc[::2, 1],data.iloc[1::2, 2],  '-', label='Markov Chain')
plt.xlabel('Sample size')
plt.ylabel('Value')
plt.legend()
plt.title('MonteCarlo vs Markov Chain (Alternate Rows)')
plt.show()

plt.figure(figsize=(15, 5))
plt.plot(data.iloc[::2, 1],data.iloc[::2, 3],  '*-', label='Error_MonteCarlo')
plt.plot(data.iloc[::2, 1],data.iloc[1::2, 3],  '-', label='Error_Markov Chain')
plt.xlabel('Sample size')
plt.ylabel('Error')
plt.legend()
plt.title('MonteCarlo vs Markov Chain (Error Comparison)')
plt.show()
