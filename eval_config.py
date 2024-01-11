from util import mkdir


# directory to store the results
results_dir = './results/'
mkdir(results_dir)

# root to the testsets
dataroot = './data_restruct_ProGAN_PNDM/test'
vals = ['PNDM_200', 'ProGAN', 'DDIM_200', 'DDPM_200', 'LDM_200', 'StyleGAN2_tmp']
multiclass = [0] * len(vals)

# model
model_path = 'weights/model_to_evaluate.pth'
