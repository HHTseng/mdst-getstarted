function convModule(inChannels, outChannels, filterSize, stride, pad)
   -- Bundle conv-norm-relu-dropout into a single module
   local conv = nn.Sequential()
   conv:add(nnlib.SpatialConvolution(inChannels, outChannels,
				  filterSize, filterSize,
				  stride, stride,
				  pad, pad))
   conv:add(nn.SpatialBatchNormalization(outChannels))
   if opt.activationFn == 'relu' then
      conv:add(nnlib.ReLU(true))
   elseif opt.activationFn == 'sigmoid' then
      conv:add(nnlib.Sigmoid())
   end
   if opt.poolingType == 'max' then
      conv:add(nnlib.SpatialMaxPooling(2, 2, 2, 2))
   elseif opt.poolingType == 'avg' then
      conv:add(nnlib.SpatialAveragePooling(2, 2, 2, 2))
   end
   conv:add(nn.Dropout(opt.dropoutRatio))
   return conv
end

function linearModule(inFeatures, outFeatures)
   -- Bundle linear-norm-relu-dropout into a single module
   local linear = nn.Sequential()   
   linear:add(nn.Linear(inFeatures, outFeatures))
   linear:add(nn.BatchNormalization(outFeatures))
   if opt.activationFn == 'relu' then
      linear:add(nnlib.ReLU(true))
   elseif opt.activationFn == 'sigmoid' then
      linear:add(nnlib.Sigmoid())
   end
   linear:add(nn.Dropout(opt.dropoutRatio))
   return linear
end
   
model = nn.Sequential()

model:add(convModule(nChannel, opt.nHidden1, opt.filterSize, 1, 2))
model:add(convModule(opt.nHidden1, opt.nHidden2, opt.filterSize, 1, 2))
model:add(convModule(opt.nHidden2, opt.nHidden3, opt.filterSize, 1, 2))

-- compute how big the feature window will be here (update this with more layers)
local nFeat = opt.nHidden3 * 5 * 5

model:add(nn.View(nFeat))
model:add(linearModule(nFeat, opt.nHidden4))
model:add(linearModule(opt.nHidden4, opt.nHidden5))
model:add(nn.Linear(opt.nHidden5, nLabel))
model:add(nnlib.Sigmoid())
model:add(nnlib.SoftMax())



if opt.gpu ~= -1 then
   model:cuda()
end
