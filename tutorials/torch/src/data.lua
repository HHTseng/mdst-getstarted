nChannel = 1

data = {}
labels = {}


dataDim = {nChannel, opt.inputRes, opt.inputRes}
labelDim = {2}

-- Load training/val labels
local labelFileTrain = io.open(opt.dataDir .. '/annot_train.txt')
io.input(labelFileTrain)
local labelsRaw = string.split(io.read(), " ")
io.close(labelFileTrain)
data['train'] = torch.Tensor(opt.nTrain, unpack(dataDim))
labels['train'] = torch.Tensor(opt.nTrain, 1):zero()
for i = 1,opt.nTrain do
   local imgName = string.format("%05d.png", annot['train']['images'][i])
   local im = image.load(opt.dataDir .. '/train/' .. imgName)
   data['train'][i] = im
   labels['train'][i] = labelsRaw[annot['train']['images'][i]+1]
end


data['valid'] = torch.Tensor(opt.nValid, unpack(dataDim))
labels['valid'] = torch.Tensor(opt.nValid, 1):zero()
for i = 1,opt.nValid do
   local imgName = string.format("%05d.png", annot['valid']['images'][i])
   local im = image.load(opt.dataDir .. '/train/' .. imgName)
   data['valid'][i] = im
   labels['valid'][i] = labelsRaw[annot['valid']['images'][i]+1]
end



data['test'] = torch.Tensor({opt.nTest, 1, opt.inputRes, opt.inputRes})




-- Define accuracy
function accuracy(output, label)
   maxs, indices = torch.max(output, 2)
   return torch.sum(torch.eq(indices, label))/output:size()[1]
end
