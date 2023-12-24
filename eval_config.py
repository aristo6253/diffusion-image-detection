from util import mkdir


# directory to store the results
results_dir = './results/'
mkdir(results_dir)

# root to the testsets
# dataroot = './dataset/test/'

# dataroot = '/scratch/izar/dimitrio/DataWang'
# vals = ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan',
#         'crn', 'imle', 'seeingdark', 'san', 'deepfake', 'stylegan2', 'whichfaceisreal']
# multiclass = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]

dataroot = './data_restruct_ProGAN_PNDM/test'
# dataroot = './data_restruct_PNDM_DDIM/test'
vals = ['PNDM_200', 'DDIM_200', 'DDPM_200', 'LDM_200', 'ProGAN']
multiclass = [0, 0, 0, 0, 0]

# model
model_path = 'weights/progan_pndm/fft_lp.pth'
