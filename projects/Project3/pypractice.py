from numpy import maximum
import torch
import logging
import os

# Configure logging to file in current directory
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.getcwd(), 'training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

dtype = torch.float
device =  torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
logger.info(f"Using {device} device")
torch.set_default_device(device)

x = torch.linspace(-1, 1, 2000, dtype=dtype) # Pick 2000 random samples from -1 to 1
y = torch.sin(x) # Taylor expansion = (1) x +(-1/6) x^3

#set up our 4 weights, setting requires_grad=true
a = torch.randn((), dtype=dtype, requires_grad=True)
b = torch.randn((), dtype=dtype, requires_grad=True)
c = torch.randn((), dtype=dtype, requires_grad=True)
d = torch.randn((), dtype=dtype, requires_grad=True)

initial_loss = 1
learning_rate = .00001
maximum_iterations = 5000

for t in range(maximum_iterations):
    logging.info(f"Current Polynomial: {a.item():10.6f}+{b.item():10.6f}x+{c.item():10.6f}x^2+{d.item():10.6f}x^3")
    # predict Y (returns a tensor x*1 big)
    y_pred = a + b * x + c * x**2 + d * x**3
    # calculate the actual loss (returns )
    loss = (y_pred - y).pow(2).sum()

    #calc initial_loss so we can see how much we have improved
    if t==0:
        initial_loss=loss.item()

    if t % 100 == 99:
        logging.info(f'Iteration t = {t:4d}  loss(t)/loss(0) = {round(loss.item()/initial_loss, 6)}')

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
    # the gradient of the loss with respect to a, b, c, d respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # Manually zero the gradients after updating weights
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
print('Taylor: (1) x +(-1/6) x^3')

