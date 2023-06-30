import os


base_dir = './chihiro'

plot_dir = os.path.join(base_dir, 'plots')
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

lambda_path = os.path.join(plot_dir, 'lambda.png')
pde_loss_path = os.path.join(plot_dir, 'pde_loss.png')
nontriv_loss_path = os.path.join(plot_dir, 'nontriv.png')
orth_loss_path = os.path.join(plot_dir, 'orth.png')
rm_path = os.path.join(plot_dir, 'rm.png')
all_path = os.path.join(plot_dir, 'all.png')
solutions_path = os.path.join(plot_dir, 'solutions.png')
