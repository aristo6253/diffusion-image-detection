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

dataroot = './data_restruct_ProGAN_hp_fft/test'
# dataroot = './data_restruct_PNDM_DDIM/test'
vals = ['DDIM_200', 'PNDM_200', 'ProGAN']
multiclass = [0, 0, 0]

# indicates if corresponding testset has multiple classes


# model
model_path = 'weights/model_epoch_best_progan_fft_hp.pth'
