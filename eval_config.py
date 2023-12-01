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

dataroot = '/scratch/izar/dimitrio/DMTest_1k_restruct'
vals = ['LDM', 'PNDM', 'StyleGAN2']
multiclass = [0, 0, 0]

# dataroot = '/scratch/izar/dimitrio/DataWang'
# vals = ['cyclegan', ]
# multiclass = [1, ]

# list of synthesis algorithms

# DATASET_PATHS = [
#     dict(
#         real_path='/scratch/izar/dimitrio/DMTestSet_1000/Real/CelebAHQ',
#         fake_path='/scratch/izar/dimitrio/DMTestSet_1000/Fake/LDM',
#         data_mode='ours',  
#         key='CelebAHQ_vs_LDM'  
#     ),
#     dict(
#         real_path='/scratch/izar/dimitrio/DMTestSet_1000/Real/CelebAHQ',
#         fake_path='/scratch/izar/dimitrio/DMTestSet_1000/Fake/PNDM',
#         data_mode='ours',  
#         key='CelebAHQ_vs_PNDM'  
#     ),
#     dict(
#         real_path='/scratch/izar/dimitrio/DMTestSet_1000/Real/CelebAHQ',
#         fake_path='/scratch/izar/dimitrio/DMTestSet_1000/Fake/StyleGAN2',
#         data_mode='ours',  
#         key='CelebAHQ_vs_StyleGAN2'   
#     ),

# ]


# Don't 

# indicates if corresponding testset has multiple classes


# model
model_path = 'weights/blur_jpg_prob0.5.pth'
