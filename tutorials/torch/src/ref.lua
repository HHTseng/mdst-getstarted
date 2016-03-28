--- Load necessary libraries and files ----------------------------------------

require 'torch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'nnx'
require 'nngraph'
require 'image'

paths.dofile('util/Logger.lua')

torch.setdefaulttensortype('torch.DoubleTensor')

-- Project directory
projectDir = '/scratch/mdatascienceteam_flux/' .. os.getenv('USER') .. '/mdst-getstarted/tutorials/torch/'

--- Process command line options ----------------------------------------------

local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)

if not opt.evalMode then
   print('Saving everything to: ' .. opt.save)
   os.execute('mkdir -p ' .. opt.save)
   torch.save(opt.save .. '/options.t7', opt)
else
   print('Loading model from: ' .. opt.save)
end


if opt.gpu == -1 then
    -- Do not use a GPU
    nnlib = nn
else
    -- Else load GPU libraries
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
    nnlib = cudnn
    cutorch.setDevice(opt.gpu) -- by default, use GPU 1
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

-- Set up training/validation splits
annot = {}

-- Randomly permute examples


if not opt.evalMode then
   local shuffle = torch.randperm(opt.nTrain)

   nTrain = torch.floor(opt.nTrain * (1 - opt.validFrac))
   nValid = torch.ceil(opt.nTrain * opt.validFrac)
   annot['train'] = shuffle:narrow(1, 1, nTrain):clone()
   annot['valid'] = shuffle:narrow(1, nTrain+1, nValid):clone()

   print('Number of training samples:   ' .. nTrain)
   print('Number of validation samples: ' .. nValid)

end

-- Setup test set
nTest = opt.nTest

annot['test'] = torch.range(1, opt.nTest)

print('Number of test samples:       ' .. nTest)


-- Set up logger
log = Logger(paths.concat(opt.save, 'train.log'), false)
