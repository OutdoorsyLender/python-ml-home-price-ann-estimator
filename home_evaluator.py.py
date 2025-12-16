# Brandon Everett
# Critical Thinking
# Module 3
# Home Price Estimator ANN

# Step 1: Collect, organize, and format the data
import numpy as np

def f(S):  # Sigmoid activation function
    return 1 / (1 + np.exp(-S))

def df_dS(S):  # Derivative of sigmoid
    return S * (1 - S)

#Example of Home Prices in Scottsdale by SQ FT
X = np.array([[1200], [1500], [1800], [2000], [2200], [2500], [2800], [3000]], dtype=np.float32)
Z = np.array([[450000], [520000], [580000], [620000], [670000], [730000], [790000], [830000]], dtype=np.float32)

# Normalize the input and output values
X = X / 3000
Z = Z / 830000

# Step 3: Decide on network architecture – 1 input, 3 hidden, 1 output
# Step 4: Select a learning algorithm – using backpropagation + sigmoid

W_input_hidden = np.random.normal(0, 1, (1, 3)) * 0.1
b_hidden = np.zeros((1, 3))
W_hidden_output = np.random.normal(0, 1, (3, 1)) * 0.1
b_output = np.zeros((1, 1))

# Step 5: Set network parameters
alpha = 0.5
epochs = 5000

# Step 6: Train the network with static backpropagation
for _ in range(epochs):
    # Forward pass
    S_hidden = np.dot(X, W_input_hidden) + b_hidden  
    Y_hidden = f(S_hidden)  

    S_output = np.dot(Y_hidden, W_hidden_output) + b_output
    Y = f(S_output)  

    # Backpropagation: compute gradients 
    error_output = Z - Y
    delta_output = error_output * df_dS(Y)

    error_hidden = delta_output.dot(W_hidden_output.T)
    delta_hidden = error_hidden * df_dS(Y_hidden)

    # Update weights and biases using gradient descent
    W_hidden_output += Y_hidden.T.dot(delta_output) * alpha
    b_output += np.sum(delta_output, axis=0, keepdims=True) * alpha

    W_input_hidden += X.T.dot(delta_hidden) * alpha
    b_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * alpha

# Step 7: Training complete – freeze weights
# Step 8: Test the trained model
print("\nEstimated Home Prices after training:")
Y_hidden_final = f(np.dot(X, W_input_hidden) + b_hidden)
Y_final = f(np.dot(Y_hidden_final, W_hidden_output) + b_output) * 830000

for i in range(len(X)):
    sqft = int(X[i][0] * 3000)
    actual = int(Z[i][0] * 830000)
    predicted = Y_final[i][0]
    print(f"{sqft} sqft → Estimated: ${predicted:,.2f} | Actual: ${actual:,.2f}")

# Step 9: Deploy the network – use it to predict unknown input
try:
    sqft_input = int(input("\nEnter square footage to estimate price of home in Scottsdale, AZ: "))
    user_X = np.array([[sqft_input / 3000]])
    hidden_user = f(np.dot(user_X, W_input_hidden) + b_hidden)
    Y_user = f(np.dot(hidden_user, W_hidden_output) + b_output)[0][0] * 830000
    print(f"Predicted price for {sqft_input} sqft home in Scottsdale: ${Y_user:,.2f}")
except ValueError:
    print("Invalid input. Please enter a number.")

