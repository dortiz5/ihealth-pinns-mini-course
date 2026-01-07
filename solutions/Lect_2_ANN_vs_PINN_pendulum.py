# -*- coding: utf-8 -*-
# Generated from Jupyter Notebook: Lect 2 - ANN Vs PINN - pendulum.ipynb
# Extracted code cells only (markdown removed).


# %% [code] Cell 2
# Import NumPy for numerical operations
import numpy as np
# Import PyTorch for building and training neural networks
import torch
import torch.nn as nn
import torch.optim as optim
# Import Matplotlib for plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

# Ignore Warning Messages
import warnings
warnings.filterwarnings("ignore")

# %% [code] Cell 3
# Setup (device + plots)
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else \
           "mps"  if torch.backends.mps.is_available() else "cpu"

def set_mpl_style(gray: str = "#5c5c5c") -> None:
    mpl.rcParams.update({
        "image.cmap": "viridis",
        "text.color": gray, "xtick.color": gray, "ytick.color": gray,
        "axes.labelcolor": gray, "axes.edgecolor": gray,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
        "font.size": 15, "interactive": False, "font.family": "sans-serif",
        "legend.loc": "best", "text.usetex": False, "mathtext.fontset": "stix",
    })

device = get_device()
print(f"Using {device} device")
set_mpl_style()

# Metrics
def add_noise(signal, snr_db):
    noise_power = np.mean(signal**2) / (10**(snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    return signal + noise, noise

def calculate_snr(signal, noise):
    signal, noise = np.asarray(signal), np.asarray(noise)
    return 10 * np.log10(np.mean(signal**2) / np.mean(noise**2))

def relative_l2_error(u_num: torch.Tensor, u_ref: torch.Tensor) -> torch.Tensor:
    return torch.norm(u_num - u_ref) / torch.norm(u_ref)

# Autodiff helper
def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """d(outputs)/d(inputs) with create_graph=True."""
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True
    )[0]

# Plotting
def plot_comparison(t: torch.Tensor, theta_true, theta_pred: torch.Tensor, loss) -> None:
    t_np = t.detach().cpu().numpy().ravel()
    pred_np = theta_pred.detach().cpu().numpy().ravel()
    true_np = np.asarray(theta_true).ravel()
    diff = np.abs(true_np - pred_np)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(t_np, true_np, label=r'$\theta(t)$ (numerical)')
    axs[0].plot(t_np, pred_np, label=r'$\theta_{\mathrm{pred}}(t)$')
    axs[0].set(title='Numerical vs Predicted', xlabel=r'Time $(s)$', 
               ylabel='Amplitude', ylim=(-1, 1.3))
    axs[0].legend(frameon=False)

    axs[1].plot(t_np, diff)
    axs[1].set(title='Absolute Difference', xlabel=r'Time $(s)$', 
               ylabel=r'$|\theta - \theta_{\mathrm{pred}}|$')
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(loss)
    ax.set(title='Training Progress', xlabel='Iteration', 
           ylabel='Loss', xscale='log', yscale='log')
    ax.grid(True)
    fig.tight_layout()
    plt.show()

# %% [code] Cell 4
from scipy.integrate import solve_ivp

# Parámetros del sistema
g, L = 9.81, 1.0 # Gravedad (m/s^2) y longitud de la varilla (m) 
theta0, omega0 = np.pi / 4, 0.0 # condiciones iniciales, ángulo (rad) y velocidad angular (rad/s) 
fs, T = 100, 10 # Frecuencia de muestreo (Hz) y tiempo total (s) 

t_eval = np.linspace(0, T, fs * T)
y0 = [theta0, omega0]

# definimos el sistema de ecuaciones diferenciales
def pendulum(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(g / L) * np.sin(theta)
    return [dtheta_dt, domega_dt]

# Solución numérica
num_sol = solve_ivp(pendulum, (0, T), y0, t_eval=t_eval, method="RK45")
theta_num, omega_num = num_sol.y

# Gráfica
plt.figure(figsize=(12, 5))
plt.plot(t_eval, theta_num, label=r'$\theta(t)$ [rad]')
plt.plot(t_eval, omega_num, label=r'$\omega(t)$ [rad/s]')
plt.xlabel('Time [s]')
plt.ylim(-2.5, 3.3)
plt.title('Nonlinear Pendulum Solution')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# %% [code] Cell 5
# Add gaussian noise
theta_noisy, noise = add_noise(theta_num, 10)  
print(f'SNR: {calculate_snr(theta_noisy, noise):.4f} dB')

t_max = 2.5           # seconds
step = 5              # downsampling factor
idx = slice(0, int(t_max * fs), step)

theta_data = theta_noisy[idx]
t_data = t_eval[idx]

# We graph the observed data
plt.figure(figsize=(12, 6))
plt.plot(t_eval, theta_num, label=r'Angular Displacement (model) $\theta(t)$ ')
plt.plot(t_data, theta_data, label=r'Training data (measures) $\theta_{data}(t)$ ')
plt.xlabel(r'Time $[s]$')
plt.ylabel(r'Angular displacement $[rad]$')
plt.ylim(-1,1.3)
plt.legend(loc='best', frameon=False)
plt.title('Training data')
plt.grid(False)
plt.show()

# %% [code] Cell 6
# Define a neural network class with user defined layers and neurons
class NeuralNetwork(nn.Module):

    def __init__(self, hlayers):
        super(NeuralNetwork, self).__init__()

        layers = []
        for i in range(len(hlayers[:-2])):
            layers.append(nn.Linear(hlayers[i], hlayers[i+1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hlayers[-2], hlayers[-1]))

        self.layers = nn.Sequential(*layers)
        self.init_params()

    def init_params(self):
        """Xavier Glorot parameter initialization of the Neural Network
        """
        def init_normal(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight) # Xavier
        self.apply(init_normal)

    def forward(self, x):
        return self.layers(x)
    

#%% Hyperparámetros para el entrenamiento
torch.manual_seed(123)
hidden_layers = [1, 50, 50, 50, 1]  # Hiperparámetros de la red ()
learning_rate = 0.001               # 
training_iter = 50000

# %% [code] Cell 7
#===============================================================================
# ETAPA 1: INFORMACIÓN DEL MODELO FÍSICO
#===============================================================================
# Numerical theta to test Numpy array to pytorch tensor
theta_test = torch.tensor(theta_num, device=device, requires_grad=True).view(-1,1).float()
# Numerical theta to train Numpy array to pytorch tensor
theta_data = torch.tensor(theta_data, device=device, requires_grad=True).view(-1,1).float()

#===============================================================================
# ETAPA 2: DEFINICIÓN DEL DOMINIO 
#===============================================================================
# Convert the NumPy arrays to PyTorch tensors and add an extra dimension
# test time Numpy array to Pytorch tensor
t_test = torch.tensor(t_eval, device=device, requires_grad=True).view(-1,1).float()
# train time Numpy array to Pytorch tensor
t_data = torch.tensor(t_data, device=device, requires_grad=True).view(-1,1).float()

#===============================================================================
# ETAPA 3: CREACIÓN DE LA RED NEURONAL SURROGANTE 
#===============================================================================
# Create an instance of the neural network
theta_ann = NeuralNetwork(hidden_layers).to(device)
nparams = sum(p.numel() for p in theta_ann.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {nparams}')

#==========================================================================
# ETAPA 5: DEFINICIÓN DE LA FUNCIÓN DE COSTO BASADA ÚNICAMENTE EN LOS DATOS
#==========================================================================
# Define a loss function (Mean Squared Error) for training the network
MSE_func = nn.MSELoss()

def NeuralNetworkLoss(forward_pass, t, theta_data, lambda1 = 1):

    theta_nn = forward_pass(t)
    data_loss = lambda1 * MSE_func(theta_nn, theta_data)

    return  data_loss

#==========================================================================
# ETAPA 6: DEFINICIÓN DEl OPTIMIZADOR
#==========================================================================
optimizer = optim.Adam(theta_ann.parameters(), lr=learning_rate,
                         betas= (0.99,0.999), eps = 1e-8)

#==========================================================================
# CICLO DE ENTRENAMIENTO
#==========================================================================
# Initialize a list to store the loss values
loss_values_ann = []

# Start the timer
start_time = time.time()

# Training the neural network
for i in range(training_iter):

    optimizer.zero_grad()   # clear gradients for next train

    # input x and predict based on x
    loss = NeuralNetworkLoss(theta_ann,
                             t_data,
                             theta_data)    # must be (1. nn output, 2. target)

    # Append the current loss value to the list
    loss_values_ann.append(loss.item())

    if i % 1000 == 0:  # print every 100 iterations
        print(f"Iteration {i}: Loss {loss.item()}")

    loss.backward() # compute gradients (backpropagation)
    optimizer.step() # update the ANN weigths

# Stop the timer and calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training time: {elapsed_time} seconds")

# %% [code] Cell 8
theta_pred_ann = theta_ann(t_test).to(device)

print(f'Relative error: {relative_l2_error(theta_pred_ann, theta_test)}')

plot_comparison(t_test, theta_num, theta_pred_ann, loss_values_ann)

# %% [code] Cell 9
#===============================================================================
# ETAPA 1: DEFINICIÓN DE LOS PARÁMETROS (MODELO FÍSICO)
#===============================================================================
# Parámetros del sistema
g, L = 9.81, 1.0 # Gravedad (m/s^2) y longitud de la varilla (m) 
theta0, omega0 = np.pi / 4, 0.0 # condiciones iniciales, ángulo (rad) y velocidad angular (rad/s) 

# Numerical theta to test Numpy array to pytorch tensor
theta_test = torch.tensor(theta_num, device=device, requires_grad=True).view(-1,1).float()
# Numerical theta to train Numpy array to pytorch tensor
theta_data = torch.tensor(theta_data, device=device, requires_grad=True).view(-1,1).float()

#===============================================================================
# ETAPA 2: DEFINICIÓN DEL DOMINIO 
#===============================================================================
# Convert the NumPy arrays to PyTorch tensors and add an extra dimension
# test time Numpy array to Pytorch tensor
t_test = torch.tensor(t_eval, device=device, requires_grad=True).view(-1,1).float()
# train time Numpy array to Pytorch tensor
t_data = torch.tensor(t_data, device=device, requires_grad=True).view(-1,1).float()

#===============================================================================
# ETAPA 3: CREACIÓN DE LA RED NEURONAL SURROGANTE 
#===============================================================================
# Create an instance of the neural network
theta_pinn = NeuralNetwork(hidden_layers).to(device)
nparams = sum(p.numel() for p in theta_pinn.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {nparams}')

#===============================================================================
# ETAPA 4 Y 5: DEFINICIÓN DE LA FUNCIÓN DE COSTO BASADA EN LA FÍSICA
#===============================================================================
# Define a loss function (Mean Squared Error) for training the network
MSE_func = nn.MSELoss()

# derivatives of the ANN
def PINNLoss(forward_pass, t_phys, t_data, theta_data, 
             lambda1 = 1, lambda2 = 1, lambda3 = 1, lambda4 = 1):

    # ANN output, first and second derivatives
    theta_pinn1 = forward_pass(t_phys)
    theta_pinn_dt = grad(theta_pinn1, t_phys)
    theta_pinn_ddt = grad(theta_pinn_dt, t_phys)
    
    f_ode = theta_pinn_ddt + (g/L) * torch.sin(theta_pinn1)
    ODE_loss = lambda1 * MSE_func(f_ode, torch.zeros_like(f_ode)) 
    
    # Define t = 0 for boundary an initial conditions
    t0 = torch.tensor(0., device=device, requires_grad=True).view(-1,1)
    
    g_ic = forward_pass(t0)
    IC_loss = lambda2 * MSE_func(g_ic, torch.ones_like(g_ic)*theta0)
    
    h_bc = grad(forward_pass(t0),t0)
    BC_loss = lambda3 * MSE_func(h_bc, torch.zeros_like(h_bc))
    
    theta_nn2 = forward_pass(t_data)
    data_loss = lambda4 * MSE_func(theta_nn2, theta_data)
    
    return ODE_loss + IC_loss + BC_loss + data_loss

#===============================================================================
# ETAPA 6: DEFINICIÓN DEl OPTIMIZADOR
#===============================================================================
# Define an optimizer (Adam) for training the network
optimizer = optim.Adam(theta_pinn.parameters(), lr=learning_rate,
                       betas= (0.99,0.999), eps = 1e-8)

#===============================================================================
# CICLO DE ENTRENAMIENTO
#===============================================================================
# Initialize a list to store the loss values
loss_values_pinn = []

# Start the timer
start_time = time.time()

# Training the neural network
for i in range(training_iter):

    optimizer.zero_grad()   # clear gradients for next train

    # input x and predict based on x
    loss = PINNLoss(theta_pinn, t_test, t_data, theta_data)

    # Append the current loss value to the list
    loss_values_pinn.append(loss.item())

    if i % 1000 == 0:  # print every 100 iterations
        print(f"Iteration {i}: Loss {loss.item()}")

    loss.backward() # compute gradients (backpropagation)
    optimizer.step() # update the ANN weigths

# Stop the timer and calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training time: {elapsed_time} seconds")

# %% [code] Cell 10
theta_pred_pinn = theta_pinn(t_test)

print(f'Relative error: {relative_l2_error(theta_pred_pinn, theta_test)}')

plot_comparison(t_test, theta_num, theta_pred_pinn, loss_values_pinn)

# %% [code] Cell 11
plot_comparison(t_test, theta_num, theta_pred_ann, loss_values_ann)
plot_comparison(t_test, theta_num, theta_pred_pinn, loss_values_pinn)
