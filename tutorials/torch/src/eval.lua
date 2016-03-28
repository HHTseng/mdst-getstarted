-- Do some helpful initialization

local testBatchSize = 4 -- opt.testBatchSize
local nBatchTest = torch.floor(nTest/testBatchSize)

print('Test batch size:    ' .. testBatchSize)

-- Load the best model
model = torch.load(paths.concat(opt.save, 'model.t7'))

-- Compute test accuracy
print(' Testing...')

testAcc = 0

-- put model in eval mode
model:evaluate()

for batch = 1,nBatchTest do

   local timer = torch.Timer()
   xlua.progress(batch, nBatchTest)

   local inputs = torch.Tensor(testBatchSize, unpack(dataDim))
   local targets = torch.LongTensor(testBatchSize, 1):zero()

   local examples = torch.range((batch-1)*testBatchSize+1, batch*testBatchSize)
   inputs, targets = loadBatch('test', examples, testBatchSize)
   if opt.gpu ~= -1 then
      inputs = inputsGPU:sub(1,testBatchSize):copy(inputs)
      targets = labelsGPU:sub(1,testBatchSize):copy(targets)
   end

   -- Forward step
   local output = model:forward(inputs)

   -- Compute accuracy
   local acc = accuracy(output, targets)
   testAcc = testAcc + acc / nBatchTest 


end

print(string.format("      Test  : Loss: --------- Acc: %.4f"  % {testAcc}))

-- Todo: logging evaluation
