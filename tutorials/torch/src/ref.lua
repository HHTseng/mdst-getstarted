--- Load necessary libraries and files ----------------------------------------

require 'torch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'nnx'
require 'nngraph'
require 'image'

--paths.dofile('img.lua')
--paths.dofile('eval.lua')

torch.setdefaulttensortype('torch.DoubleTensor')

-- Project directory
projectDir = os.getenv('HOME') .. '/mdst-getstarted/tutorials/torch/'

--- Process command line options ----------------------------------------------

local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)

--print('Saving everything to: ' .. opt.save)
--os.execute('mkdir -p ' .. opt.save)

if opt.GPU == -1 then
    -- Do not use a GPU
    nnlib = nn
else
    -- Else load GPU libraries
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
    nnlib = cudnn
    cutorch.setDevice(opt.GPU) -- by default, use GPU 1
end

-- Training hyperparameters
if not optimState then
    optimState = {
        learningRate = opt.LR,
        learningRateDecay = opt.LRdecay,
        momentum = opt.momentum,
        dampening = 0.0,
        weightDecay = opt.weightDecay
    }
end
if opt.optMethod == 'adadelta' then optfn = optim.adadelta
elseif opt.optMethod == 'sgd' then optfn = optim.sgd
elseif opt.optMethod == 'rmsprop' then optfn = optim.rmsprop
elseif opt.optMethod == 'nag' then optfn = optim.nag end

-- Random number generation seed
if opt.manualSeed ~= -1 then torch.manualSeed(opt.manualSeed)
else torch.seed() end

--- Load in annotations -------------------------------------------------------

-- Load in annotations
annotLabels = {'train', 'valid', 'test'}
annot = {}

annot['train'] = {}
annot['train']['nsamples'] = opt.nTrain
annot['train']['images'] = {}
local idx = 1
for i = 1,opt.nTrain do
    annot['train']['images'][i] = idx-1
    idx = idx + 1
end

annot['valid'] = {}
annot['valid']['nsamples'] = opt.nValid
annot['valid']['images'] = {}
for i = 1,opt.nValid do
    annot['valid']['images'][i] = idx-1
    idx = idx + 1
end

annot['test'] = {}
annot['test']['nsamples'] = opt.nTest
annot['test']['images'] = {}
for idx = 1,opt.nTest do
    annot['test']['images'][idx] = idx-1
end


-- Default input is assumed to be an image and output is assumed to be a heatmap
-- This can change if an hdf5 file is used, or if opt.task specifies something different
dataDim = {1, opt.inputRes, opt.inputRes}
