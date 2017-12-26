
require 'paths'
require 'image'
require 'tool'

torch.manualSeed(2018)

prepareDataset = {}
local DatasetGenerator = {}
DatasetGenerator.__index = DatasetGenerator
setmetatable(DatasetGenerator, {
    __call = function(cls, ...)
        return cls.new(...)
    end,
})


function DatasetGenerator.new(video_id_image, modelname, geipath)
    local self = setmetatable({}, DatasetGenerator)
    if is_grey == nil then
        is_grey = false
    end
    self.geipath = geipath
    
    local to_number_start = 1
    self.hid2index = {}
    self.index2hid = {}
    self.item2imagefilenames = {}
    self.item_data = {}
    for line in io.lines(video_id_image) do
        local res = splitByComma(line)
        local item = res[1]
        local hid_end, _ = string.find(item, '-')
        local hid = string.sub(item, 1, hid_end - 1)
        if self.hid2index[hid] == nil then
            self.hid2index[hid] = to_number_start
            table.insert(self.index2hid, hid)
            to_number_start = to_number_start + 1
        end
        self.item2imagefilenames[item] = {}
        table.insert(self.item_data, item)
        for i = 2, #res do
            table.insert(self.item2imagefilenames[item], res[i])
        end
    end
    
    -- set variable for gei_next_batch()
    self.gei_index = 1
    self.epoch = 1
    self.item_retrieve_indexs = torch.randperm(#self.item_data)
    self.width = 88
    self.height = 128
    
    return self
end

function DatasetGenerator:next_batch(size)
    local targets = {}
    local inputs = {}
    local infos = {}
    local batch = self:next_gei_pair_batch_oulp(size)
    for i = 1, #batch do
        local video1 = batch[i][1]
        local video2 = batch[i][2]
        local hid1 = batch[i][3]
        local hid2 = batch[i][4]
        local pos = batch[i][5]
        local item1 = batch[i][6]
        local item2 = batch[i][7]
        local target
        if pos then
            target = 2
        else
            target = 1
        end
        table.insert(inputs, {video1, video2})
        table.insert(targets, target)
        table.insert(infos, {item1, item2})
    end
    return inputs, targets, infos
end

function DatasetGenerator:load_gei(video_id)
    local img_paths = self.item2imagefilenames[video_id]
    local endremove_example = "//00000097.png"
    local preremove_example = "/home/chenqiang/data/gait-rnn/../OULP_C1V1_Pack//OULP-C1V1_NormalizedSilhouette(88x128)//"
    local keep_example = "Seq00/6324539"
    local start_i = string.len(preremove_example)
    local end_i = start_i + string.len(keep_example)
    local keep_part = string.sub(img_paths[1], start_i, end_i)
    local final_filename = string.format("%s/%s.png", self.geipath, keep_part)
    local img = image.load(final_filename, 1)
    return img
end


function DatasetGenerator:next_gei_batch(batch_size)
    local i = self.gei_index
    local res = {}
    while #res ~= batch_size do
        if i > #self.item_data then
            info('reset self.gei_index to 1, %02d epoch ends', self.epoch)
            self.item_retrieve_indexs = torch.randperm(#self.item_data)
            self.epoch = self.epoch + 1
            i = 1
        end
        index = self.item_retrieve_indexs[i]
        local item = self.item_data[index]
        local hid_number = self:get_hid_number(item)
        local gei = self:load_gei(item)
        table.insert(res, {gei, hid_number})
        i = i + 1
    end
    self.gei_index = i
    
    
    -- reshape the res
    local geis = torch.zeros(batch_size, 1, self.height, self.width)
    local targets = {}
    for i=1, #res do
        geis[{{i}}] = res[i][1]
        table.insert(targets, res[i][2])
    end
    
    return geis, targets
end

function DatasetGenerator:get_hid_number(item)
    local hid_end, _ = string.find(item, '-')
    local hid = string.sub(item, 1, hid_end - 1)
    local hid_number = self.hid2index[hid]
    return hid_number
end

function DatasetGenerator:_next_pair_batch_oulp_item(is_pos)
    local seqs = {'Probe', 'Gallery' }
    local views = {55, 65, 75, 85 }
    local item_tpl = "%s-IDList_OULP-C1V1-A-%s_%s.csv"
    local item1, item2
    if is_pos then
        local hid_idx = torch.random(1, #self.index2hid)
        local s_i1 = torch.random(1, #seqs)
        local s_i2 = torch.random(1, #seqs - 1)

        local v_i1 = torch.random(1, #views)
        local v_i2 = torch.random(1, #views - 1)

        if s_i2 >= s_i1 then
            s_i2 = s_i2 + 1
        end

        if v_i2 >= v_i1 then
            v_i2 = v_i1 + 1
        end

        local hid = self.index2hid[hid_idx]
        item1 = string.format(item_tpl, hid, views[v_i1], seqs[s_i1])
        item2 = string.format(item_tpl, hid, views[v_i2], seqs[s_i2])
    else
        local hid1_idx = torch.random(1, #self.index2hid)
        local hid2_idx = torch.random(1, #self.index2hid - 1)
        if hid2_idx >= hid1_idx then
            hid2_idx = hid2_idx + 1
        end
        local s_i1 = torch.random(1, #seqs)
        local s_i2 = torch.random(1, #seqs)
        local v_i1 = torch.random(1, #views)
        local v_i2 = torch.random(1, #views)
        local hid1 = self.index2hid[hid1_idx]
        local hid2 = self.index2hid[hid2_idx]
        item1 = string.format(item_tpl, hid1, views[v_i1], seqs[s_i1])
        item2 = string.format(item_tpl, hid2, views[v_i2], seqs[s_i2])
    end
    return item1, item2

end

function DatasetGenerator:next_gei_pair_batch_oulp(batch_size)
    local pos = true
    local pos_batch = self:_next_gei_pair_batch_oulp(batch_size/2, pos)
    local neg_batch = self:_next_gei_pair_batch_oulp(batch_size/2, not pos)
    local res = {}
    for i=1, #pos_batch do
        table.insert(res, pos_batch[i])
        table.insert(res, neg_batch[i])
    end
    return res
end


function DatasetGenerator:_next_gei_pair_batch_oulp(batch_size, is_pos)
    local res = {}
    local item1, item2
    while #res ~= batch_size do
        item1, item2 = self:_next_pair_batch_oulp_item(is_pos)
        local item1_number = self:get_hid_number(item1)
        local item2_number = self:get_hid_number(item2)
        local images1 = self:load_gei(item1)
        local images2 = self:load_gei(item2)
        table.insert(res, {images1, images2, item1_number, item2_number, is_pos, item1, item2})
    end
    return res
end


function prepareDataset.prepareDatasetOULP(datapath, modelname, geipath)
    local train_filename = string.format('%s/oulp_train_data.txt', datapath)
    local test_filename = string.format('%s/oulp_test_data.txt', datapath)
    local val_filename = string.format('%s/oulp_val_data.txt', datapath)
    local res = {}
    res['train'] = DatasetGenerator.new(train_filename, modelname, geipath)
    res['val'] = DatasetGenerator.new(val_filename, modelname, geipath)
    res['test'] = DatasetGenerator.new(test_filename, modelname, geipath)
    info(string.format('load data from %s, %s, %s', train_filename, val_filename, test_filename))
    return res
end
return prepareDataset