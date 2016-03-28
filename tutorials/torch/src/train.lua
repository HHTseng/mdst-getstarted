-- Do some helpful initialization

local batchSize = opt.batchSize
local testBatchSize = 4 -- opt.testBatchSize

local nBatchTrain = torch.floor(nTrain/batchSize)
local nBatchValid = torch.floor(nValid/testBatchSize)

local criterion = nn.CrossEntropyCriterion()
if opt.gpu ~= -1 then
   criterion:cuda()
end

print('Training batch size: ' .. batchSize)
print('Valid batch size:    ' .. testBatchSize)

local bestValidAcc = -1

for epoch = 0,opt.nEpochs do

   print('Epoch ' .. epoch)
   local param, gradparam = model:getParameters()
   
   local trainErr = 0
   local trainAcc = 0
   local validErr = 0
   local validAcc = 0

   if opt.gpu ~= -1 then cutorch.synchronize() end
   collectgarbage()
   
   -- back to training mode
   model:training()
   local shuffle = torch.randperm(nTrain)   
   -- Perform training iteration

   if epoch > 0 then
      print(' Training...')
      for batch = 1,nBatchTrain do

	 local timer = torch.Timer()
	 xlua.progress(batch, nBatchTrain)

	 local examples = shuffle:narrow(1, (batch-1)*batchSize+1, batchSize)
	 inputs, targets = loadBatch('train', examples, batchSize)
	 if opt.gpu ~= -1 then
	    inputs = inputsGPU:sub(1,batchSize):copy(inputs)
	    targets = labelsGPU:sub(1,batchSize):copy(targets)
	 end

	 -- Forward step
	 local output = model:forward(inputs)
	 local err = criterion:forward(output, targets)

	 -- Compute loss
	 trainErr = trainErr + err / nBatchTrain

	 -- Backward step
	 model:zeroGradParameters()
	 model:backward(inputs, criterion:backward(output, targets))

	 local function evalFn(x) return err, gradparam end
	 optfn(evalFn, param, optimState)

	 -- Compute accuracy
	 local acc = accuracy(output, targets)
	 trainAcc = trainAcc + acc / nBatchTrain 

      end
   end


   -- Compute validation accuracy
   print(' Validation...')
   
   -- put model in eval mode
   model:evaluate()
   local shuffle = torch.randperm(nValid)   
   for batch = 1,nBatchValid do

      local timer2 = torch.Timer()
      xlua.progress(batch, nBatchValid)

      local inputs = torch.Tensor(testBatchSize, unpack(dataDim))
      local targets = torch.LongTensor(testBatchSize, 1):zero()

      local examples = shuffle:narrow(1, (batch-1)*testBatchSize+1, testBatchSize)
      inputs, targets = loadBatch('valid', examples, testBatchSize)
      if opt.gpu ~= -1 then
         inputs = inputsGPU:sub(1,testBatchSize):copy(inputs)
	 targets = labelsGPU:sub(1,testBatchSize):copy(targets)
      end

      -- Forward step
      local output = model:forward(inputs)
      local err = criterion:forward(output, targets)

      -- Compute loss
      validErr = validErr + err / nBatchValid


      -- Compute accuracy
      local acc = accuracy(output, targets)
      validAcc = validAcc + acc / nBatchValid 
   end

   print(string.format("      Train : Loss: %.7f Acc: %.4f"  % {trainErr, trainAcc}))
   print(string.format("      Valid : Loss: %.7f Acc: %.4f"  % {validErr, validAcc}))

   if log then
        log:add{
            ['epoch     '] = string.format("%d" % epoch),
            ['train loss'] = string.format("%.6f" % trainErr),
            ['train acc '] = string.format("%.4f" % trainAcc),
            ['valid loss'] = string.format("%.6f" % validErr),
            ['valid acc '] = string.format("%.4f" % validAcc),
            ['LR        '] = string.format("%g" % optimState.learningRate)
        }
    end
    
    if validAcc > bestValidAcc then
       bestValidAcc = validAcc
       print('      Saving...')
       torch.save(paths.concat(opt.save, 'model.t7'), model)
       torch.save(paths.concat(opt.save, 'optimState.t7'), optimState)
    end

   
end

