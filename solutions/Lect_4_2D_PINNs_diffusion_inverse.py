# %%
# NumPy para operaciones numéricas
import numpy as np
# PyTorch para construir y entrenar redes neuronales
import torch
import torch.nn as nn
import torch.optim as optim
# Matplotlib para graficar
import matplotlib.pyplot as plt
import matplotlib as mpl
# Time para medir tiempo de entrenamiento
import time
# Warnings para ignorar mensajes de advertencia
import warnings
warnings.filterwarnings("ignore")

from matplotlib import animation, rc
from scipy.stats import qmc
# %%
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
    
# Definir pi en torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2

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

def plot_comparison(u_true, u_pred, loss, k_evol):
    u_hat = u_pred.detach().cpu().numpy()

    # --- Soluciones ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for a, u, title in zip(
        ax,
        [u_true, u_hat],
        ['Analytic solution', 'PINN solution']
    ):
        im = a.imshow(u, extent=[-1, 1, 2, 0])
        a.set(title=title, xlabel=r'$x$', ylabel=r'$t$')
        fig.colorbar(im, ax=a, shrink=0.5)

    fig.tight_layout()
    plt.show()

    # --- Entrenamiento ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(k_evol, label='PINN estimate')
    ax[0].hlines(1, 0, len(k_evol), label='True value')
    ax[0].set(title=r'$\kappa$ evolution', xlabel='Iteration')
    ax[0].legend(frameon=False)

    ax[1].plot(loss)
    ax[1].set(title='Training Progress',
              xlabel='Iteration', ylabel='Loss',
              xscale='log', yscale='log')
    ax[1].grid(True)

    fig.tight_layout()
    plt.show()
    
def animate(x,t,U):
    # Primero, generar figura con subplot correspondiente
    fig, ax = plt.subplots(figsize = (10,6))
    plt.close()
    ax.set_title(r"heat solution $e^t sin(\pi x)$")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(np.floor(U.min()), np.ceil(U.max()))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x, t)$")
    # Inicializar etiqueta sin texto
    time_label = ax.text(1, 1, "", color = "black", fontsize = 12)
    # Inicializar gráfico sin datos
    line, = ax.plot([], [], color = "black", lw = 2)
    
    # Definir función de inicialización
    def init():
        line.set_data([], [])
        time_label.set_text("")
        return (line,)

    # Animar función. Esta función se llama secuencialmente con FuncAnimation
    def animate(i):
        line.set_data(x, U[i])
        time_label.set_text(f"t = {t[i]:.2f}")
        return (line,)

    return animation.FuncAnimation(fig, animate, init_func=init,
                                frames=len(t), interval=50, blit=True)
# %%
# Número de muestras para espacio y tiempo.
dom_samples = 100

# Función para solución analítica
def analytic_diffusion(x,t):
    u = np.exp(-t)*np.sin(np.pi*x)
    return u

# Dominio espacial
x = np.linspace(-1, 1, dom_samples)
# Dominio temporal
t = np.linspace(0, 2, dom_samples)

# Mallado
X, T = np.meshgrid(x, t)
# Evaluar función en mallado
U = analytic_diffusion(X, T)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, T, U, cmap='viridis', edgecolor='k')

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(t, x)')
ax.set_title('3D Analytic Solution for Diffusion')

# Añadir la barra de color
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

plt.show()
# %%
# Correr animación y hacer display
anim = animate(x,t,U)
rc("animation", html="jshtml")
anim
# %%
from scipy.stats import qmc
# Muestreo con LHS
sampler = qmc.LatinHypercube(d=2)
sample = sampler.random(n=100)

# Límites del dominio
l_bounds = [-1, 0]
u_bounds = [ 1, 2]
domain_xt = qmc.scale(sample, l_bounds, u_bounds)

# Tensores de torch
x_ten = torch.tensor(domain_xt[:, 0], requires_grad = True).float().reshape(-1,1)
t_ten = torch.tensor(domain_xt[:, 1], requires_grad = True).float().reshape(-1,1)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(domain_xt[:, 0],domain_xt[:, 1], label = 'PDE collocation points')
ax.set_title('Collocation points')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$t$')
ax.legend(loc='lower left')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
# %%
# Evaluar puntos en función analítica
x_np = x_ten.detach().numpy()
t_np = t_ten.detach().numpy()
u_true = analytic_diffusion(x_np,t_np).reshape(1, -1)
u_observ = u_true + np.random.normal(0,0.01,len(x_np))
# Convertir observaciones a torch
u_observ_t = torch.tensor(u_observ, requires_grad = True).float().reshape(-1,1)
# %%
# Definir clase de red neuronal con capas y neuronas especificadas por usuario
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
        """Inicialización de parámetros Xavier Glorot
        """
        def init_normal(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight) # Xavier
        self.apply(init_normal)

    def forward(self, x):
        return self.layers(x)
    
#===============================================================================
# ETAPA 3: CREACIÓN DE LA RED NEURONAL SURROGANTE 
#===============================================================================
torch.manual_seed(123)

# hiper-parámetros de la red
hidden_layers = [2, 10, 10, 10, 1]

# Crear instancia de la NN
u_pinn = NeuralNetwork(hidden_layers)
nparams = sum(p.numel() for p in u_pinn.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {nparams}')

#==========================================================================
# ETAPA 6: DEFINICIÓN DEl OPTIMIZADOR
#==========================================================================
learning_rate = 0.001

# Tratar k como parámetro entrenable
kappa = torch.nn.Parameter(torch.ones(1, requires_grad=True)*2)
kappas = []

# Definir optimizador y agregar k
optimizer = optim.Adam(list(u_pinn.parameters())+[kappa], lr=0.001,
                       betas= (0.99,0.999), eps = 1e-8)
# %%
#===============================================================================
# ETAPA 1: DEFINICIÓN DE LOS PARÁMETROS (MODELO FÍSICO)
#===============================================================================
from scipy.stats import qmc
# Muestreo con LHS
sampler = qmc.LatinHypercube(d=2)
sample = sampler.random(n=100)

# Límites del dominio
l_bounds = [-1, 0]
u_bounds = [ 1, 2]
domain_xt = qmc.scale(sample, l_bounds, u_bounds)

# Función para solución analítica
def analytic_diffusion(x,t):
    u = np.exp(-t)*np.sin(np.pi*x)
    return u

# Evaluar puntos en función analítica
x_np = domain_xt[:, 0]
t_np = domain_xt[:, 1]
u_true = analytic_diffusion(x_np,t_np).reshape(1, -1)
u_observ = u_true + np.random.normal(0,0.01,len(x_np))
# Convertir observaciones a torch
u_observ_t = torch.tensor(u_observ, requires_grad = True).float().reshape(-1,1)

#===============================================================================
# ETAPA 2: DEFINICIÓN DEL DOMINIO 
#===============================================================================
# Tensores de torch
x_ten = torch.tensor(domain_xt[:, 0], requires_grad = True).float().reshape(-1,1)
t_ten = torch.tensor(domain_xt[:, 1], requires_grad = True).float().reshape(-1,1)

#===============================================================================
# ETAPA 3: CREACIÓN DE LA RED NEURONAL SURROGANTE 
#===============================================================================
torch.manual_seed(123)

# hiper-parámetros de la red
hidden_layers = [2, 10, 10, 10, 1]

# Crear instancia de la NN
u_pinn = NeuralNetwork(hidden_layers)
nparams = sum(p.numel() for p in u_pinn.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {nparams}')

#==========================================================================
# ETAPA 4 Y 5: DEFINICIÓN DE LA FUNCIÓN DE COSTO BASADA EN AUTOGRAD
#==========================================================================
# Error cuadrático medio (Mean Squared Error - MSE)
MSE_func = nn.MSELoss()

def PINN_diffusion_Loss(forward_pass, x_ten, t_ten, kappa,
             lambda1 = 1, lambda2 = 1):

    # ANN output, first and second derivatives
    domain = torch.cat([t_ten, x_ten], dim = 1)
    u = forward_pass(domain)
    u_t = grad(u, t_ten)
    u_x = grad(u, x_ten)
    u_xx = grad(u_x, x_ten)
    
    # PDE loss definition
    f_pde = u_t - kappa*u_xx + torch.exp(-t_ten)*(torch.sin(np.pi*x_ten) 
                                        -(torch.pi**2)*torch.sin(np.pi*x_ten))
    PDE_loss = lambda1 * MSE_func(f_pde, torch.zeros_like(f_pde)) 
    
    # Data loss
    data_loss = lambda2 * MSE_func(u, u_observ_t)
    
    return PDE_loss + data_loss

#==========================================================================
# ETAPA 6: DEFINICIÓN DEl OPTIMIZADOR
#==========================================================================
learning_rate = 0.001

# Tratar k como parámetro entrenable
kappa = torch.nn.Parameter(torch.ones(1, requires_grad=True)*2)
kappas = []

# Definir optimizador y agregar k
optimizer = optim.Adam(list(u_pinn.parameters())+[kappa], lr=0.001,
                       betas= (0.99,0.999), eps = 1e-8)


#==========================================================================
# CICLO DE ENTRENAMIENTO
#==========================================================================

training_iter = 25000

# Inicializar lista para guardar valores de pérdida
loss_values = []

# Empezar timer
start_time = time.time()

# Entrenar red neuronal
for i in range(training_iter):

    optimizer.zero_grad()   # Reinicializar gradientes para iteración de entrenamiento

    # ingresar x, predecir con PINN y obtener pérdida
    loss = PINN_diffusion_Loss(u_pinn, x_ten, t_ten, kappa)

    # Agregar actual valor de pérdida a la lista y agregar valor de kappa actual
    loss_values.append(loss.item())
    kappas.append(kappa.item())

    if i % 1000 == 0:  # Imprimir pérdida cada 1000 iteraciones
        print(f"Iteration {i}: Loss {loss.item()}")

    loss.backward() # Paso de retropropagación
    optimizer.step() # Actualizar pesos de la red con optimizador

# Detener timer y obtener tiempo transcurrido
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training time: {elapsed_time} seconds")
# %%
dom_samples = 100

# Dominio espacial
x = np.linspace(-1, 1, dom_samples)
# Dominio temporal
t = np.linspace(0, 2, dom_samples)

# Mallado
X, T = np.meshgrid(x, t)

# Evaluar función en mallado
U = analytic_diffusion(X, T)

X_ten = torch.tensor(X).float().reshape(-1, 1)
T_ten = torch.tensor(T).float().reshape(-1, 1)
domain_ten = torch.cat([T_ten, X_ten], dim = 1)
U_pred = u_pinn(domain_ten).reshape(dom_samples,dom_samples)

U_true = torch.tensor(U).float()
print(f'Relative error: {relative_l2_error(U_pred, U_true)}')

plot_comparison(U, U_pred, loss_values, kappas)