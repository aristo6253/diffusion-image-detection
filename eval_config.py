from util import mkdir


# directory to store the results
results_dir = './results/'
mkdir(results_dir)

# root to the testsets
dataroot = './data_restruct_ProGAN_PNDM/test'
vals = ['PNDM_200', 'DDIM_200', 'DDPM_200', 'LDM_200', 'ProGAN']
multiclass = [0, 0, 0, 0, 0]

# model
model_path = 'weights/fft_2.pth'
