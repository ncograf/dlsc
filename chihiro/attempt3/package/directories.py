import os


base_dir = './chihiro/attempt3'

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

model_dir = os.path.join(base_dir, 'models')
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
model1_path = os.path.join(model_dir, 'model1.pth')
model2_path = os.path.join(model_dir, 'model2.pth')
model3_path = os.path.join(model_dir, 'model3.pth')
model4_path = os.path.join(model_dir, 'model4.pth')
