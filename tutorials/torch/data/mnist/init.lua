-- Read MNIST data for torch

local nTrain = 60000
local nTest = 10000

local dataDim = {1, 32, 32}
nLabel = 10

nChannel = 1

-- Load training examples

local trainFile = opt.dataDir .. '/mnist.t7/train_32x32.t7'
local trn = torch.load(trainFile, 'ascii')

data['train'] = torch.Tensor(opt.nTrain, unpack(dataDim))
labels['train'] = torch.LongStorage(opt.nTrain)

for i = 1,opt.nTrain do
   local idx = annot['train']['images'][i]+1
   data['train'][i] = trn['data'][idx]:clone()
   labels['train'][i] = trn['labels'][idx]
end

data['valid'] = torch.Tensor(opt.nTrain, unpack(dataDim))
labels['valid'] = torch.LongStorage(opt.nTrain)

for i = 1,opt.nValid do
   local idx = annot['valid']['images'][i]+1
   data['valid'][i] = trn['data'][idx]:clone()
   labels['valid'][i] = trn['labels'][idx]
end





function preprocess(inputs, targets)
   -- Preprocess batch (e.g. cropping, rescaling, etc)

   return inputs, targets
end

