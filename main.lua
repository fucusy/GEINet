require 'test'

cmd = torch.CmdLine()
cmd:option('-iteration', 30,'how many iteration')
cmd:option('-gradclip',5,'magnitude of clip on the RNN gradient')
cmd:option('-modelname','GEINet','wuzifeng model name you want to load')
cmd:option('-dropout',0.5,'fraction of dropout to use between layers')
cmd:option('-learningrate',1e-3)
cmd:option('-datapath', '/home/chenqiang/data/gait-rnn', 'base data path')
cmd:option('-geipath', '/home/chenqiang/data/OULP_C1V1_Pack_GEI/')
cmd:option('-calprecision', 200, 'calculate loss on validation every batch')
cmd:option('-calval', 2, 'calculate loss on validation every batch')
cmd:option('-batchsize', 64, 'how many intance in a traning batch')
cmd:option('-loadmodel', '', 'load fullmodel, rnn model, cnn model')
cmd:option('-gpu', false, 'use GPU')
cmd:option('-gpudevice', 1, 'set gpu device')
arg = arg or ""
opt = cmd:parse(arg)
opt.gpu = true
print(opt)

-- set the GPU
if opt.gpu then
    require 'cunn'
    require 'cutorch'
    cutorch.setDevice(opt.gpudevice)
end

require 'model.getModel';
model, crit = GEINet(opt.gpu, opt.dropout)


local prepDataset = require 'prepareDataset'
dataset = prepDataset.prepareDatasetOULP(opt.datapath, opt.modelname, opt.geipath)
for i, item in ipairs({'train', 'val', 'test'}) do
    local item_count = #dataset[item].index2hid
    local uniq_count = #dataset[item].item_data
    info('%s data instances %05d, uniq  %04d', item, item_count, uniq_count)
end


local parameters, gradParameters = model:getParameters()
info('Number of parameters:%d', parameters:size(1))
local max_val_precision = 0

local timer = torch.Timer()
local i = 0
while i < opt.iteration do
    i = i + 1    
    
    -- calculate loss on validation dataset, and test dataset for testing purpose
--     if i % opt.calval == 0 then
--         local val_in, val_tar = dataset['val']:next_gei_batch(opt.batchsize)
--         val_in = convertToCuda(val_in)
--         local val_loss = cal_loss(model, crit, val_in, val_tar)
--         info('%05dth/%05d Val Error %0.6f', i, opt.iteration, val_loss)
        
--         local tes_in, tes_tar = dataset['test']:next_gei_batch(opt.batchsize)
--         tes_in = convertToCuda(tes_in)
--         local loss = cal_loss(model, crit, tes_in, tes_tar)            
--         info('%05dth/%05d Tes Error %0.6f', i, opt.iteration, loss)
--     end
    

    local inputs, targets = dataset['train']:next_gei_batch(opt.batchsize)
    gradParametersAdd = gradParameters:clone()
    gradParametersAdd:zero()
    total_loss = 0            
    inputs = convertToCuda(inputs)
    for j = 1, opt.batchsize do
        inputPair = inputs[j]
        output = model:forward(inputPair)
        loss = crit:forward(output, targets[j])
        total_loss = total_loss + loss
        local grad = crit:backward(output, targets[j])        
        gradParameters:zero()
        model:backward(inputPair, grad)
        gradParameters:clamp(-opt.gradclip, opt.gradclip)
        gradParametersAdd:add(gradParameters)
    end
    total_loss = total_loss / opt.batchsize
    gradParametersAdd:div(opt.batchsize)
    parameters:add(gradParametersAdd*opt.learningrate*-1)   
    if i % opt.calprecision == 0 then
        local same, diff, prec = evaluate_oulp_simi(dataset['val'], model)
        if prec > max_val_precision then
            info('change max precision from %0.2f to %0.2f'
                                        , max_val_precision, prec)
            max_val_precision = prec
            local name = string.format('%s_valpre_%0.04f_i%04d', opt.modelname, max_val_precision, i)
            save_model(model, name)
        else
            info('do not change max_precision from %0.2f to %0.2f', max_val_precision, prec)
        end
    end
    
    
    local time = timer:time().real
    timer:reset()
    info('%05dth/%05d Tra Error %0.6f, %d', i, opt.iteration, total_loss, time)
end