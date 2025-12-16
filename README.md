# python-ml-home-price-ann-estimator

This repository contains a Python-based Artificial Neural Network (ANN) project that estimates residential home prices based on square footage.
The model was built from scratch using NumPy only, without high-level ML libraries, to demonstrate a clear understanding of neural network fundamentals.

This project was created as part of coursework for CSC506 and serves as an introductory machine-learning portfolio piece.

🧠 Project Overview

The neural network uses:

One input node (square footage)

One hidden layer with three neurons

One output node (estimated home price)

The model is trained using backpropagation and a sigmoid activation function to learn pricing trends from a small dataset of Scottsdale, AZ home prices.

After training, the network can:

Display predicted vs actual home prices

Accept user input to estimate the price of an unknown home

📁 Files Included

Everett_HomeValuator_Module_3.py
Main Python script containing:

Data normalization

Network architecture

Training loop

Backpropagation logic

Interactive user prediction

🚀 How to Run the Project

Install Python 3

Install NumPy if not already installed:

pip install numpy


Run the script:

python Everett_HomeValuator_Module_3.py


Follow the prompt to enter a square-footage value and receive a price estimate.

🛠️ Technologies Used

Python 3

NumPy

Manual implementation of:

Sigmoid activation function

Gradient descent

Backpropagation

🎯 Learning Objectives Demonstrated

This project demonstrates understanding of:

Neural network architecture

Forward and backward propagation

Activation functions

Data normalization

Training loops and convergence

Applying ML concepts to a real-world problem

📌 Notes

The dataset is intentionally small and simplified for learning purposes

Pricing data is illustrative and not intended for real-world valuation use

No external ML frameworks were used to emphasize conceptual understanding

## 👤 Author

**Brandon Everett**  
ML & AI Graduate Student  
Outdoorsy Lender
