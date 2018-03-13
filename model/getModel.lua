require 'nn'
function GEINet(gpu, dropout)
    local input_h = 128
    local input_w = 88
    local input_channel = 1
    local conv_kernel = {18, 45}
    local conv_size = {7, 5}
    local conv_tride = {1, 1}
    local pool_size = {2, 3}
    local pool_tride = {2, 2}
    local LRN_size = 5
    local LRN_alpha = 0.0001
    local LRN_beta = 0.75
    local LRN_k = 2
    local ConvNet
    if gpu then
        ConvNet = nn.SpatialConvolutionMM
    else
        ConvNet = nn.SpatialConvolution
    end

    local layer1 = nn.Sequential()
    layer1:add(ConvNet(input_channel, conv_kernel[1], conv_size[1], conv_size[1], conv_tride[1], conv_tride[1]))
    layer1:add(nn.ReLU())
    layer1:add(nn.SpatialMaxPooling(pool_size[1], pool_size[1], pool_tride[1], pool_tride[1]))
    layer1:add(nn.SpatialCrossMapLRN(LRN_size, LRN_alpha, LRN_beta, LRN_k))
    
    local layer2 = nn.Sequential()    
    layer2:add(ConvNet(conv_kernel[1], conv_kernel[2], conv_size[2], conv_size[2], conv_tride[2], conv_tride[2]))
    layer2:add(nn.ReLU())
    layer2:add(nn.SpatialMaxPooling(pool_size[2], pool_size[2], pool_tride[2], pool_tride[2]))
    layer2:add(nn.SpatialCrossMapLRN(LRN_size, LRN_alpha, LRN_beta, LRN_k))

    local fc = nn.Sequential()
    local previous_count = 45*18*28
    fc:add(nn.Reshape(previous_count))
    fc:add(nn.Dropout(dropout))
    fc:add(nn.Linear(previous_count, 1024))
    fc:add(nn.ReLU())
    fc:add(nn.Linear(1024, 856))
 
    
    local model = nn.Sequential()
    model:add(layer1)
    model:add(layer2)
    model:add(fc)
    model:add(nn.LogSoftMax())
    
    if gpu then
        model = model:cuda()
    end
    local crit = nn.ClassNLLCriterion()
    if gpu then
        crit = crit:cuda()
    end
    return model, crit
end