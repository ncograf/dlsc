import os
import __main__

base_dir = os.path.dirname(os.path.realpath(__main__.__file__))

plot_dir = os.path.join(base_dir, 'plots')
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

lambda_path = os.path.join(plot_dir, 'lambda.png')
pde_loss_path = os.path.join(plot_dir, 'pde_loss.png')
nontriv_loss_path = os.path.join(plot_dir, 'nontriv.png')
orth_loss_path = os.path.join(plot_dir, 'orth.png')
rm_path = os.path.join(plot_dir, 'rm.png')
all_path = os.path.join(plot_dir, 'all.png')
all_solutions_path = os.path.join(plot_dir, 'all_solutions.png')
solutions_path = os.path.join(plot_dir, 'solutions.png')

model_dir = os.path.join(base_dir, 'models')
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
model_path = []
for i in range(8):
    model_path.append(os.path.join(model_dir, f'model{i}.pth'))