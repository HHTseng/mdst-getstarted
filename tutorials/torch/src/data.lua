-- Initialize default global vars
nChannel = 1
data = {}
labels = {}
dataDim = {nChannel, opt.inputRes, opt.inputRes}
labelDim = {2}

-- Load dataset and related functions

dofile(opt.dataDir .. '/init.lua')

-- Load training/val labels
--[[
local labelFileTrain = io.open(opt.dataDir .. '/annot_train.txt')
io.input(labelFileTrain)
local labelsRaw = string.split(io.read(), " ")
io.close(labelFileTrain)
data['train'] = torch.Tensor(opt.nTrain, unpack(dataDim))
labels['train'] = torch.LongStorage(opt.nTrain)
for i = 1,opt.nTrain do
   local imgName = string.format("%05d.png", annot['train']['images'][i])
   local im = image.load(opt.dataDir .. '/train/' .. imgName)
   data['train'][i] = im
   labels['train'][i] = labelsRaw[annot['train']['images'][i]+1]
end

data['valid'] = torch.Tensor(opt.nValid, unpack(dataDim))
labels['valid'] = torch.LongTensor(opt.nValid, 1)
for i = 1,opt.nValid do
   local imgName = string.format("%05d.png", annot['valid']['images'][i])
   local im = image.load(opt.dataDir .. '/train/' .. imgName)
   data['valid'][i] = im
   labels['valid'][i] = labelsRaw[annot['valid']['images'][i]+1]
end

data['test'] = torch.Tensor({opt.nTest, 1, opt.inputRes, opt.inputRes})
--]]



-- Preallocate memory for fast GPU
if opt.gpu ~= -1 then
    local bs = math.max(opt.batchSize,opt.testBatchSize)
    inputsGPU = torch.CudaTensor(bs, unpack(dataDim))
    labelsGPU = torch.CudaTensor(bs, 1)
end

-- Define some useful functions

function loadBatch(set, idxs)
   -- Returns a single minibatch of data with indices idxs

   local batchSize = idxs:size()[1]
   local inputs = torch.Tensor(batchSize, unpack(dataDim))
   local targets = torch.LongTensor(batchSize,1)
   for example = 1,batchSize do
      local exampleIdx = idxs[example]
      inputs[example] = data[set][exampleIdx]:clone()
      targets[example] = labels[set][exampleIdx]
   end
   return preprocess(inputs, targets) -- preprocess batch if needed
end

function accuracy(output, label)
   maxs, indices = torch.max(output, 2)
   return torch.sum(torch.eq(indices, label))/output:size()[1]
end
