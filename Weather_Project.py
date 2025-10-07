import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

#Dataset Upload
url = 'https://raw.githubusercontent.com/Devanshu-04/4375-Project/main/dataset/oslo_weather_blocks.csv'
wea = pd.read_csv(url)

# new data columns
temp_cols = ['day1_temp', 'day2_temp', 'day3_temp', 'day4_temp', 'day5_temp', 'y_day6_temp']
humid_cols = ['day1_humid', 'day2_humid', 'day3_humid', 'day4_humid', 'day5_humid']
wind_cols = ['day1_windspd', 'day2_windspd', 'day3_windspd', 'day4_windspd', 'day5_windspd']

# minmax scaling (normalizing data)
temp_values = wea[temp_cols].values
min_temp = temp_values.min()
max_temp = temp_values.max()
wea.loc[:, temp_cols] = (wea[temp_cols] - min_temp) / (max_temp - min_temp)

humid_values = wea[humid_cols].values
min_humid = humid_values.min()
max_humid = humid_values.max()
wea.loc[:, humid_cols] = (wea[humid_cols] - min_humid) / (max_humid - min_humid)

wind_values = wea[wind_cols].values
min_wind = wind_values.min()
max_wind = wind_values.max()
wea.loc[:, wind_cols] = (wea[wind_cols] - min_wind) / (max_wind - min_wind)

# Perform train test split
X = wea.drop(columns=['y_day6_temp', 'y_change'])
y = wea[['y_day6_temp']]

# Classification scrapped
# y = wea[['y_day6_temp', 'y_change']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=76518571)

# Classification scrapped, updates made to achieve target deadline
# def numpyify(X, y):
#     X = X.values.reshape(-1, 5, 3)
#     y_reg = y['y_day6_temp'].values.reshape(-1, 1)
#     y_cls = y['y_change'].values.reshape(-1, 1)
#     return X, y_reg, y_cls
# X_train, y_train_reg, y_train_cls = numpyify(X_train, y_train)
# X_test, y_test_reg, y_test_cls = numpyify(X_test, y_test)

# convert to model format
def numpyify(X, y):
    X = X.values.reshape(-1, 5, 3)
    y_reg = y.values.reshape(-1, 1)
    return X, y_reg

X_train, y_train_reg = numpyify(X_train, y_train)
X_test, y_test_reg  = numpyify(X_test, y_test)

#Activation functions
def tanh(x):
  return np.tanh(x)

def tanh_derivative(x):
  return 1.0 - np.tanh(x)**2

def relu(x):
  return np.maximum(0, x)

def relu_derivative(x):
  return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

activation_functions = {
    "tanh": (tanh, tanh_derivative),
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative)
}

#RNN Implementation
class RNN:
  def __init__(self, input_size, hidden_layer_sizes=(10,), activation = "sigmoid", reg_lambda= 0, learning_rate = 0.01, epochs = 100):
    #These initialize the rnn with hyperparameters and weight matrixes
    self.input_size = input_size
    self.hidden_layer_sizes = hidden_layer_sizes
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.reg_lambda = reg_lambda
    self.activation, self.activation_derivative = activation_functions[activation]
    self.init_weights()

  def init_weights(self):
    #code below is in regards to single rnn cells where: input node -> hidden node
    #hidden node -> hidden node
    hidden_size = self.hidden_layer_sizes[0]
    self.W_input_hidden = np.random.randn(hidden_size, self.input_size) * 0.01
    self.W_hidden_hidden = np.random.randn(hidden_size, hidden_size) * 0.01
    self.bias_hidden = np.zeros((hidden_size,1))

    #output
    self.W_hidden_regression = np.random.randn(1, hidden_size) * 0.01
    self.bias_regression = np.zeros((1,1))

    # Classification scrapped
    # self.W_hidden_classification = np.random.randn(1, hidden_size) * 0.01
    # self.bias_classification = np.zeros((1,1))

  def forward_pass(self, sequence):

    #Will perform forward pass through the rnn
    #the hidden state from the previous time step (h_previous) is used to calc.
    #hidden state of the next state. Helping maintain a memory of sorts of the past information.

    h_previous = np.zeros((self.hidden_layer_sizes[0], 1))
    hidden_states = []
    z_values = []

    for time_step in range(sequence.shape[0]):
      x_t = sequence[time_step].reshape(-1, 1)
      z_t = np.dot(self.W_input_hidden, x_t) + np.dot(self.W_hidden_hidden, h_previous) + self.bias_hidden
      h_t = self.activation(z_t)

      hidden_states.append(h_t)
      z_values.append(z_t)
      h_previous = h_t

    #input coming from last hidden node
    y_regression = np.dot(self.W_hidden_regression, h_previous) + self.bias_regression
    return hidden_states, z_values, y_regression

    # Classification scrapped
    # y_classification = sigmoid(np.dot(self.W_hidden_classification, h_previous) + self.bias_classification)
    # return hidden_states, z_values, y_regression, y_classification

 # TODO: Classification scrapped
 # def compute_gradients(self, sequence, hidden_states, z_values, y_regression, y_classification, target_regression, target_classification):
  def compute_gradients(self, sequence, hidden_states, z_values, y_regression, target_regression):
    #Computes the gradient using BPTT
    #this is designed for a regression model, classification was scrapped due to
    #errors occuring during training. The model can focus on temperature prediction now.


    #initialize
    dW_input_hidden = np.zeros_like(self.W_input_hidden)
    dW_hidden_hidden = np.zeros_like(self.W_hidden_hidden)
    db_hidden = np.zeros_like(self.bias_hidden)

    dW_hidden_regression = np.zeros_like(self.W_hidden_regression)
    db_regression = np.zeros_like(self.bias_regression)

    # Classification scrapped
    # dW_hidden_classification = np.zeros_like(self.W_hidden_classification)
    # db_classification = np.zeros_like(self.bias_classification)

    #output
    error_reg = 2 * (y_regression - target_regression)
    dW_hidden_regression += np.dot(error_reg, hidden_states[-1].T)
    db_regression += error_reg

    # Classification scrapped
    # error_cls = (y_classification - target_classification) * sigmoid_derivative(y_classification)
    # dW_hidden_classification += np.dot(error_cls, hidden_states[-1].T)
    # db_classification += error_cls
    # dh_next = np.dot(self.W_hidden_regression.T, error_reg) + np.dot(self.W_hidden_classification.T, error_cls)

    # derivatives up to hidden layer
    dh_next = np.dot(self.W_hidden_regression.T, error_reg)

    for time_step in reversed(range(len(sequence))):
      dz = dh_next * self.activation_derivative(z_values[time_step])
      x_t = sequence[time_step].reshape(-1, 1)
      h_prev = hidden_states[time_step - 1] if time_step > 0 else np.zeros_like(hidden_states[0])
      dW_input_hidden += np.dot(dz, x_t.T)
      dW_hidden_hidden += np.dot(dz, h_prev.T)
      db_hidden += dz
      dh_next = np.dot(self.W_hidden_hidden.T, dz)

      # L2 weight updates
    if self.reg_lambda > 0:
      dW_input_hidden += self.reg_lambda * self.W_input_hidden
      dW_hidden_hidden += self.reg_lambda * self.W_hidden_hidden
      dW_hidden_regression += self.reg_lambda * self.W_hidden_regression

      # Classification scrapped
      # dW_hidden_classification += self.reg_lambda * self.W_hidden_classification

    gradients = (dW_input_hidden, dW_hidden_hidden, db_hidden, dW_hidden_regression, db_regression)
    return gradients

    # Classification scrapped
    # gradients = (dW_input_hidden, dW_hidden_hidden, db_hidden, dW_hidden_regression, db_regression, dW_hidden_classification, db_classification)


  def update_weights(self, gradients):
    # Classification scrapped
    # (dW_input_hidden, dW_hidden_hidden, db_hidden, dW_hidden_regression, db_regression, dW_hidden_classification, db_classification) = gradients

    # using gradient descent
    (dW_input_hidden, dW_hidden_hidden, db_hidden, dW_hidden_regression, db_regression) = gradients

    self.W_input_hidden -= self.learning_rate * dW_input_hidden
    self.W_hidden_hidden -= self.learning_rate * dW_hidden_hidden
    self.bias_hidden -= self.learning_rate * db_hidden

    self.W_hidden_regression -= self.learning_rate * dW_hidden_regression
    self.bias_regression -= self.learning_rate * db_regression

    # Classification scrapped
    # self.W_hidden_classification -= self.learning_rate * dW_hidden_classification
    # self.bias_classification -= self.learning_rate * db_classification

  # Classification scrapped
  # def train(self, X, y_reg, y_cls):
  def train(self, X, y_reg):
    for epoch in range(self.epochs):
      predictions_reg = []
      total_loss = 0

      # Classification scrapped
      # predictions_cls = []

      for i in range(X.shape[0]):
        seq = X[i]
        target_r = y_reg[i]

        # Classification scrapped
        # target_c = y_cls[i]

        if isinstance(target_r, np.ndarray) and target_r.size == 1:
          target_r = target_r.item()
        # Classification scrapped
        # if isinstance(target_c, np.ndarray) and target_c.size == 1:
        #   target_c = target_c.item()

        # Classification scrapped
        # hs, zs, y_r, y_c = self.forward_pass(seq)

        hs, zs, y_r = self.forward_pass(seq)

        if isinstance(y_r, np.ndarray) and y_r.size == 1:
          y_r = y_r.item()
        # Classification scrapped
        # if isinstance(y_c, np.ndarray) and y_c.size == 1:
        #   y_c = y_c.item()

        # Loss (MSE + L2)
        # mse_loss = mean_squared_error([target_r], y_r.flatten())
        mse_loss = mean_squared_error([target_r], [y_r])

        # Classification scrapped
        # # Old Loss (MSE + BCE + L2)
        # bce_loss = -(target_c * np.log(y_c + 1e-8) + (1 - target_c) * np.log(1 - y_c + 1e-8))

        l2_loss = 0
        if self.reg_lambda > 0:
            l2_loss += np.sum(self.W_input_hidden**2)
            l2_loss += np.sum(self.W_hidden_hidden**2)
            l2_loss += np.sum(self.W_hidden_regression**2)

            # Classification scrapped
            # l2_loss += np.sum(self.W_hidden_classification**2)

        total_loss += mse_loss + (self.reg_lambda * l2_loss / 2)

        gradients = self.compute_gradients(seq, hs, zs, y_r, target_r)
        self.update_weights(gradients)

        # Classification scrapped
        # total_loss += mse_loss + bce_loss + (self.reg_lambda * l2_loss / 2)
        # gradients = self.compute_gradients(seq, hs, zs, y_r, y_c, target_r, target_c)

        predictions_reg.append(y_r)

        # Classification scrapped
        # predictions_cls.append(int(y_c > 0.5))

      mse = mean_squared_error(y_reg, predictions_reg)

      # Classification scrapped
      # acc = accuracy_score(y_cls, predictions_cls)

      # print(type(mse), mse.shape if hasattr(mse, 'shape') else 'no shape')
      # print(type(acc), acc.shape if hasattr(acc, 'shape') else 'no shape')
      # print(type(total_loss), total_loss.shape if hasattr(total_loss, 'shape') else 'no shape')
      # print(f"Epoch {epoch+1}/{self.epochs} | MSE = {mse:.4f} | Acc= {acc:.4f} | Total loss = {float(total_loss):.4f}")
      print(f"Epoch {epoch+1}/{self.epochs} | MSE = {mse:.4f} | Total loss = {float(total_loss):.4f}")

# Model is probably ready for training, go ahead.
rnn = RNN(input_size=3, hidden_layer_sizes=(25,), activation='tanh', learning_rate=0.1, epochs=1000)
rnn.train(X_train, y_train_reg)

# Classification scrapped
# def evaluate(X, y_reg, y_cls, rnn):
def evaluate(X, y_reg, rnn):
    y_pred_reg = []

    # Classification scrapped
    # y_pred_cls = []

    for i in range(X.shape[0]):
        _, _, y_r = rnn.forward_pass(X[i])
        y_pred_reg.append(float(y_r.item()))

        # Classification scrapped
        # _, _, y_r, y_c = rnn.forward_pass(X[i])
        # y_pred_cls.append(int(y_c > 0.5))

    y_reg = y_reg.reshape(-1)

    print("Final MSE:", round(mean_squared_error(y_reg, y_pred_reg), 4))
    print("Final RMSE:", round(root_mean_squared_error(y_reg, y_pred_reg),4))
    print("Final MAE:", round(mean_absolute_error(y_reg, y_pred_reg), 4))
    print("Final R² Score:", round(r2_score(y_reg, y_pred_reg), 4))
    print("Final ExplVar: ", round(explained_variance_score(y_reg, y_pred_reg), 4))

    # Classification scrapped
    # y_cls = y_cls.reshape(-1)
    # print("Final Accuracy:", round(accuracy_score(y_cls, y_pred_cls),4))
    # print("Classification Report:\n", classification_report(y_cls, y_pred_cls, zero_division=0))
    # print(confusion_matrix(y_cls, y_pred_cls))

# train evaluate
print("")
print("Training:")
evaluate(X_train, y_train_reg, rnn)
print("")
# test evaluate
print("Testing:")
evaluate(X_test, y_test_reg, rnn)
