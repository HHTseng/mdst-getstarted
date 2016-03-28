-- Read MNIST data for torch

local dataDim = {1, 32, 32}
nLabel = 10
nChannel = 1

-- Load training examples

if not opt.evalMode then

   local trainFile = opt.dataDir .. '/mnist.t7/train_32x32.t7'
   local trn = torch.load(trainFile, 'ascii')

   data['train'] = torch.Tensor(nTrain, unpack(dataDim))
   labels['train'] = torch.LongStorage(nTrain)

   for i = 1,nTrain do
      local idx = annot['train'][i]
      data['train'][i] = trn['data'][idx]:clone()
      labels['train'][i] = trn['labels'][idx]
   end

   data['valid'] = torch.Tensor(nValid, unpack(dataDim))
   labels['valid'] = torch.LongStorage(nValid)

   for i = 1,nValid do
      local idx = annot['valid'][i]
      data['valid'][i] = trn['data'][idx]:clone()
      labels['valid'][i] = trn['labels'][idx]
   end

end

local testFile = opt.dataDir .. '/mnist.t7/test_32x32.t7'
local tst = torch.load(testFile, 'ascii')

data['test'] = torch.Tensor(nTest, unpack(dataDim))
labels['test'] = torch.LongStorage(nTest)

for i = 1,nTest do
   local idx = annot['test'][i]
   data['test'][i] = tst['data'][idx]:clone()
   labels['test'][i] = tst['labels'][idx]
end





function preprocess(inputs, targets)
   -- Preprocess batch (e.g. cropping, rescaling, etc)

   return inputs, targets
end

