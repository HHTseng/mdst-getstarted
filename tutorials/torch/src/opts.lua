--projectDir = '/scratch/jiadeng_flux/stroud/mdst-getstarted/tutorials/torch/'

local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text(' ---------- General options ------------------------------------')
    cmd:text()
    cmd:option('-dataDir',  projectDir .. '/data', 'Data directory')
    cmd:option('-expDir',    projectDir .. '/exp', 'Experiment directory')
    cmd:option('-dataset',                'mnist', 'Dataset choice')
    cmd:option('-name',                 'myModel', 'Name of model to save/load') 
    cmd:option('-evalMode',                 false, 'Skip training and go straight to eval')
    cmd:option('-manualSeed',                  -1, 'Manually set RNG seed')
    cmd:text()
    cmd:text(' ---------- Model options --------------------------------------')
    cmd:text()
    cmd:option('-filterSize',          3, 'Filter size')
    cmd:option('-poolingType',     'max', 'Pooling type (max | avg)')
    cmd:option('-activationFn',   'relu', 'Activation function (relu | sigmoid)')
    cmd:option('-nHidden1',           64, 'Number of channels, conv layer 1')
    cmd:option('-nHidden2',           64, 'Number of channels, conv layer 2')
    cmd:option('-nHidden3',           32, 'Number of channels, conv layer 3')
    cmd:option('-nHidden4',         4096, 'Number of channels, fc layer 4')
    cmd:option('-nHidden5',         1000, 'Number of channels, fc layer 5')
    cmd:text()
    cmd:text(' ---------- Hyperparameter options -----------------------------')
    cmd:text()
    cmd:option('-dropoutRatio',      0.5, 'Dropout ratio')
    cmd:option('-LR',             2.5e-4, 'Base learning rate')
    cmd:option('-LRdecay',           0.0, 'Learning rate decay')
    cmd:option('-momentum',         0.95, 'Momentum')
    cmd:option('-weightDecay',      1e-5, 'Weight decay')
    cmd:option('-optMethod',   'rmsprop', 'Optimization method: (sgd | nag | adadelta | rmsprop)')
    cmd:text()
    cmd:text(' ---------- Training options -----------------------------------')
    cmd:text()
    cmd:option('-nEpochs',           100, 'Total number of epochs to run')
    cmd:option('-batchSize',          64, 'Mini-batch size (1 = pure stochastic)')
    cmd:text()
    cmd:text(' ---------- Testing options ------------------------------------')
    cmd:text()
    cmd:option('-testBatchSize',       4, 'Mini-batch size for testing/validation (currently fixed at 4)')
    cmd:text()
    cmd:text(' ---------- Data options ---------------------------------------')
    cmd:text()
    cmd:option('-inputRes',           32, 'Input image resolution (currently unused)')
    cmd:option('-nTrain',             -1, 'Number of training samples (-1 to use all)')
    cmd:option('-validFrac',         0.2, 'Fraction of training samples to use for validation')
    cmd:option('-nTest',              -1, 'Number of test samples (-1 to use all)')
    cmd:text()
    cmd:text(' ---------- GPU options ----------------------------------------')
    cmd:text()
    cmd:option('-gpu',                -1, 'GPU to use (-1 for no GPU)')
    
    local opt = cmd:parse(arg or {})
    opt.dataDir = paths.concat(opt.dataDir, opt.dataset)
    opt.expDir = paths.concat(opt.expDir, opt.dataset)
    opt.save = paths.concat(opt.expDir, opt.name)

    if opt.nTrain == -1 then
       opt.nTrain = false
    end
    return opt

end

return M
