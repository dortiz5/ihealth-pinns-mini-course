# Extracted from: Lect 1 - PINN Mass - spring- damper.ipynb
# Code cells only (includes inline code comments).


# %% [cell 2]
# Import NumPy for numerical operations
import numpy as np
# Import PyTorch for building and training neural networks
import torch
import torch.nn as nn
import torch.optim as optim
# Import Matplotlib for plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
# Import the time module to time our training process
import time
# Ignore Warning Messages
import warnings
warnings.filterwarnings("ignore")

# %% [cell 3]
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
def relative_l2_error(u_num: torch.Tensor, u_ref: torch.Tensor) -> torch.Tensor:
    return torch.norm(u_num - u_ref) / torch.norm(u_ref)

# Util function to plot the solutions
def plot_comparison(t, theta_true, theta_pred, loss):
    t, u, u_hat = (
        x.detach().cpu().numpy().ravel()
        for x in (t, theta_true, theta_pred)
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(t, u, label=r'$\theta(t)$ (numerical)')
    ax[0].plot(t, u_hat, label=r'$\theta_{\mathrm{pred}}(t)$')
    ax[0].set(title='Numerical vs Predicted',
              xlabel=r'Time $(s)$', ylabel='Amplitude', ylim=(-1, 1.3))
    ax[0].legend(frameon=False)

    ax[1].plot(t, np.abs(u - u_hat))
    ax[1].set(title='Absolute Difference',
              xlabel=r'Time $(s)$',
              ylabel=r'$|\theta - \theta_{\mathrm{pred}}|$')

    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(loss)
    ax.set(title='Training Progress',
           xlabel='Iteration', ylabel='Loss',
           xscale='log', yscale='log')
    ax.grid(True)
    fig.tight_layout()
    plt.show()

# %% [cell 4]
# Dominio temporal
T = 5.0        # tiempo total de simulación
x0 = 1.0       # Posición inicial 
v0 = 0.0       # velocidad incial
wn = 5.0       # Frecuencia natural
zeta = 0.2     # razón de amortiguamiento

# %% [cell 5]
def crear_dominio_temporal(T, N_train=101, N_eval=1000):
    """Crea el dominio temporal para la PINN."""
    t_train = torch.linspace(0, T, N_train, 
                             device=device, 
                             requires_grad=True).view(-1, 1)  # entrenamiento
    t_eval = torch.linspace(0, T, N_eval, 
                             device=device, 
                             requires_grad=True).view(-1, 1)  # evaluación
    return t_train, t_eval # dominio de evaluación

# %% [cell 6]
# Define a neural network class with user defined layers and neurons
class NeuralNetwork(nn.Module):

    def __init__(self, hlayers = [1, 10, 10, 1]):
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

# %% [cell 7]
# Util function to calculate tensor gradients with autodiff
def grad(outputs, inputs):
    """Computes the partial derivative of an output with respect
    to an input.
    Args:
        outputs: (N, 1) tensor
        inputs: (N, D) tensor
    """
    return torch.autograd.grad(outputs, inputs,
                        grad_outputs=torch.ones_like(outputs),
                        create_graph=True,
                        )[0]


# Definir tensor de entrada. Si queremos derivar c/r a x necesitamos inicializar con requires_grad=True
x = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True).view(-1,1).float() # (N,1)

# Calcular operación que dependen de x
y = x**2 # (N,1)  

# Calcular derivadas c/r a x 
# grad es un wrapper de torch.autograd
dy_dx = grad(y, x) 

# Calcular derivadas de orden superior
d2y_dx2 = grad(dy_dx, x)  

print("x:", x)
print("y = x^2:", y)
print("dy/dx:", dy_dx)
print("d^2y/dx^2:", d2y_dx2)  

# Esto también funciona para redes neuronales
# test_ANN = NeuralNetwork()

# NNx = test_ANN(x)
# dNNx_dx = grad(NNx, x)
# print("NNx: ", dNNx_dx)
# print("dNNx/dx:", dNNx_dx)

# %% [cell 8]
# Define a loss function (Mean Squared Error) for training the network
MSE_func = nn.MSELoss()

# derivatives of the ANN
def PINNLoss(PINN, t_phys, wn, zeta, x0 = 1, v0 = 0,
             lambda1 = 1, lambda2 = 1, lambda3 = 1):

    t0 = torch.tensor(0., device=device, requires_grad=True).view(-1,1)

    # ANN output, first and second derivatives
    x_pinn_t = PINN(t_phys)
    x_pinn_dt = grad(x_pinn_t, t_phys)
    x_pinn_ddt = grad(x_pinn_dt, t_phys)
    
    f_ode = x_pinn_ddt + 2 * zeta * wn * x_pinn_dt + wn**2 * x_pinn_t
    ODE_loss = lambda1 * MSE_func(f_ode, torch.zeros_like(f_ode)) 
    
    g_ic = PINN(t0)
    IC_loss = lambda2 * MSE_func(g_ic, torch.ones_like(g_ic)*x0)
    
    h_bc = grad(PINN(t0),t0)
    BC_loss = lambda3 * MSE_func(h_bc, torch.ones_like(h_bc)*v0)
    
    return ODE_loss + IC_loss + BC_loss

# %% [cell 9]
def pinn_optimizer(pinn, lr = 0.01):

    # Define an optimizer (Adam) for training the network
    return optim.Adam(pinn.parameters(), lr=lr,
                        betas= (0.99,0.999), eps = 1e-8)

# %% [cell 10]
#===============================================================================
# ETAPA 1: INFORMACIÓN DEL MODELO FÍSICO
#===============================================================================
# Dominio temporal
T = 5.0        # tiempo total de simulación
x0 = 1.0       # Posición inicial 
v0 = 0.0       # velocidad incial
wn = 5.0  # Frecuencia natural
zeta = 0.2     # razón de amortiguamiento

#===============================================================================
# ETAPA 2: DEFINICIÓN DEL DOMINIO 
#===============================================================================
# Creamos los tensores de tiempo para el entrenamiento y la evaluación
t_train, t_eval = crear_dominio_temporal(T)

#===============================================================================
# ETAPA 3: CREACIÓN DE LA RED NEURONAL SURROGANTE 
#===============================================================================
# Creamos la ANN
torch.manual_seed(123)
hidden_layers = [1, 30, 30, 30, 1]# Parámetros de la 

# Create an instance of the neural network
x_pinn = NeuralNetwork(hidden_layers).to(device)
nparams = sum(p.numel() for p in x_pinn.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {nparams}')

#==========================================================================
# ETAPA 4 Y 5: DEFINICIÓN DE LA FUNCIÓN DE COSTO BASADA EN LA FÍSICA
#==========================================================================
# Define a loss function (Mean Squared Error) for training the network
MSE_func = nn.MSELoss()

# derivatives of the ANN
def PINNLoss(PINN, t_phys, wn, zeta, x0 = 1, v0 = 0, 
             w1 = 1, w2 = 1, w3 = 1):

    t0 = torch.tensor(0., device=device, requires_grad=True).view(-1,1)

    # ANN output, first and second derivatives
    x_pinn_t = PINN(t_phys)
    x_pinn_dt = grad(x_pinn_t, t_phys)
    x_pinn_ddt = grad(x_pinn_dt, t_phys)
    
    f_ode = x_pinn_ddt + 2 * zeta * wn * x_pinn_dt + wn**2 * x_pinn_t
    ODE_loss = w1 * MSE_func(f_ode, torch.zeros_like(f_ode)) 
    
    g_ic = PINN(t0)
    IC_loss = w2 * MSE_func(g_ic, torch.ones_like(g_ic)*x0)
    
    h_bc = grad(PINN(t0),t0)
    BC_loss = w3 * MSE_func(h_bc, torch.zeros_like(h_bc)*v0)
    
    return ODE_loss + IC_loss + BC_loss 

#==========================================================================
# ETAPA 6: DEFINICIÓN DEl OPTIMIZADOR
#==========================================================================
learning_rate = 0.01
optimizer = pinn_optimizer(x_pinn, learning_rate)

#==========================================================================
# CICLO DE ENTRENAMIENTO
#==========================================================================
training_iter = 20000

# Initialize a list to store the loss values
loss_values_pinn = []

# Start the timer
start_time = time.time()

# Training the neural network
for i in range(training_iter):

    optimizer.zero_grad()   # clear gradients for next train

    # input x and predict based on x
    loss = PINNLoss(x_pinn, t_train, wn, zeta)

    # Append the current loss value to the list
    loss_values_pinn.append(loss.item())

    if i % 500 == 0:  # print every 100 iterations
        print(f"Iteration {i}: Loss {loss.item()}")

    loss.backward() # compute gradients (backpropagation)
    optimizer.step() # update the ANN weigths

# Stop the timer and calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training time: {elapsed_time} seconds")

# %% [cell 11]
def masa_resorte_general(t, x0, v0, omega_n, zeta):
    """
    Solución exacta x(t) del sistema masa-resorte-amortiguador:
        x'' + 2*zeta*omega_n*x' + omega_n^2*x = 0
    Incluye los tres regímenes (sub, crítico y sobreamortiguado).
    """
    t = np.array(t, dtype=float)

    if 0 < zeta < 1:  # Subamortiguado
        omega_d = omega_n * np.sqrt(1 - zeta**2)
        x = np.exp(-zeta * omega_n * t) * (
            x0 * np.cos(omega_d * t) +
            (v0 + zeta * omega_n * x0) / omega_d * np.sin(omega_d * t)
        )

    else:
        raise ValueError("zeta debe ser mayor que 0.")

    return torch.tensor(x, dtype=torch.float32, device=device).view(-1, 1)

# -------------------------------------------
# solucion exacta
t_eval_np = t_eval.detach().cpu().numpy().ravel()
x = masa_resorte_general(t_eval_np, x0, v0, wn, zeta)

# predicción de la PINN
x_pred_pinn = x_pinn(t_eval)

print(f'Relative error: {relative_l2_error(x_pred_pinn, x)}')

plot_comparison(t_eval, x, x_pred_pinn, loss_values_pinn)
