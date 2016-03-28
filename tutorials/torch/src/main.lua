paths.dofile('ref.lua')     -- Parse command line input and global variable initialization
paths.dofile('data.lua')    -- Set up our data framework
paths.dofile('model.lua')   -- Read in network model

if not opt.evalMode then
   paths.dofile('train.lua')
end

paths.dofile('eval.lua')
