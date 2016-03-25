local nTrain = annot['train']['nsamples']
local nValid = annot['valid']['nsamples']
local batchSize = opt.batchSize
local testBatchSize = 4 -- opt.testBatchSize
local criterion = nn.CrossEntropyCriterion()

for epoch = 1,opt.nEpochs do

   print('Epoch ' .. epoch)
   local param, gradparam = model:getParameters()
   

   local trainErr = 0
   local trainAcc = 0

   local validAcc = 0
   
   -- Compute validation accuracy
   -- put model in eval mode
   model:evaluate()
   local shuffle = torch.randperm(nValid)   
   for batch = 1,torch.floor(nValid/testBatchSize) do
      local inputs = torch.Tensor(testBatchSize, unpack(dataDim))
      local targets = torch.LongTensor(testBatchSize, 1):zero()
      local examples = torch.range((batch-1)*testBatchSize+1, batch*testBatchSize)
      for example = 1,testBatchSize do
	 local exampleIdx = shuffle[examples[example]]
	 local im = data['valid'][exampleIdx]:clone()
	 inputs[example] = im
	 local target = labels['valid'][exampleIdx]:clone()
	 targets[example] = target+1
      end
      -- Forward step
      local output = model:forward(inputs)
      -- Compute accuracy
      local acc = accuracy(output, targets)
      validAcc = validAcc + acc / torch.floor(nValid/testBatchSize) 
   end

   print('Validation Accuracy: ' .. validAcc)

   -- back to training mode
   model:training()
   local shuffle = torch.randperm(nTrain)   
   -- Perform training iteration
   for batch = 1,torch.floor(nTrain/batchSize) do

      print('Batch ' .. batch)
      local inputs = torch.Tensor(batchSize, unpack(dataDim))
      local targets = torch.LongTensor(batchSize, 1):zero()
      local examples = torch.range((batch-1)*batchSize+1,batch*batchSize)
      for example = 1,batchSize do
	 local exampleIdx = shuffle[examples[example]]
	 local im = data['train'][exampleIdx]:clone()
	 inputs[example] = im
	 local target = labels['train'][exampleIdx]:clone()
	 targets[example] = target+1
      end

      -- Forward step
      local output = model:forward(inputs)
      local err = criterion:forward(output, targets)

      -- Compute loss
      trainErr = trainErr + err / torch.floor(nTrain/batchSize)

      -- Backward step
      model:zeroGradParameters()
      model:backward(inputs, criterion:backward(output, targets))

      local function evalFn(x) return err, gradparam end
      optfn(evalFn, param, optimState)

      -- Compute accuracy
      local acc = accuracy(output, targets)
      trainAcc = trainAcc + acc / torch.floor(nTrain/batchSize) 
         
   end

   print('Train Accuracy: ' .. trainAcc)

   
end

