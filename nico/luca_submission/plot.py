from main5 import Pinns

if __name__ == "__main__":
    x0, xf = -6., 6.
    epochs = 4
    n_samples = 2000
    batch_size = n_samples
    neurons = 50

    pinn = Pinns(neurons, n_samples, batch_size, x0, xf, load1 = True, load2 = True, load3 = True, load4 = True)
    
    pinn.plot_all()

