require 'tool'

function evaluate_oulp_simi(dataset, model, n_persons_force)
    --[[
    -- for gallery, probe, 55, 65, 75, 85
    -- extract feature by model classfication model, to extract 8 sequences feature for every one
    -- calculate distance between every one in probe in some view with every one in gallery in some view
    -- have to calculate (n * 8) * (n * 8) where n is the person number times, if n = 100, then 640000 times compares
    --]]


    local n_persons = n_persons_force or #dataset.index2hid
    
    local avg_same = 0
    local avg_diff = 0
    local sum_precision = 0
    local avg_same_count = 0
    local avg_dif_count = 0
    local g_p_views = {55, 65, 75, 85}
    local eer_str = {}

    -- sim_mat '{gallery_view, '%02d'}-{probe_view, '%02d'}' => torch.zeros(n_persons,n_persons), first dim is probe idx,
    -- second dim is gallery idx
    -- prepare gallery view for all persons
    local gallery_feat_tbl = {}
    local probe_feat_tbl = {}
    local feature_extract = feature_extract_model(model)
    local merge = merge_model()
    if opt ~= nil and opt.gpu then
        merge = merge:cuda()
    end
    for _, view in ipairs(g_p_views) do
        local gallery_feats = {}
        local probe_feats = {}
        for i=1, n_persons do
            local full_item_name =
                        string.format('%s-IDList_OULP-C1V1-A-%02d_Gallery.csv'
                                    ,dataset.index2hid[i], view)

            local full_item_name_probe = 
                         string.format('%s-IDList_OULP-C1V1-A-%02d_Probe.csv'
                                        ,dataset.index2hid[i], view)
            local imgs = dataset:load_gei(full_item_name)
            local imgs_probe = dataset:load_gei(full_item_name_probe)
            if opt ~= nill and opt.gpu then
                imgs = imgs:cuda()
                imgs_probe = imgs_probe:cuda()
            end            
            
            local output = clone_value(feature_extract:forward(imgs))
            table.insert(gallery_feats, output)
            local p_output = clone_value(feature_extract:forward(imgs_probe))
            table.insert(probe_feats, p_output)
        end
        gallery_feat_tbl[view] = gallery_feats
        probe_feat_tbl[view] = probe_feats
    end
    -- sim_mat '{gallery_view, '%02d'}-{probe_view, '%02d'}' => torch.zeros(n_persons,n_persons), first dim is probe idx,
    -- second dim is gallery idx
    local sim_mat = {}
    for _, probe_view in ipairs(g_p_views) do
        for _, gallery_view in ipairs(g_p_views) do
            local pro = {}
            local res = {}
            for i = 1,n_persons do
                for gallery_j = 1, n_persons do
                    local gallery_name = dataset.index2hid[gallery_j]
                    local probe_name = dataset.index2hid[i]
                    local gallery_item = string.format(
                                         '%s-IDList_OULP-C1V1-A-%s_Gallery.csv'
                                            , gallery_name, gallery_view)
                    local probe_item = string.format(
                                        '%s-IDList_OULP-C1V1-A-%s_Probe.csv'
                                        , probe_name, probe_view)
                    local key_str = string.format('%02d-%02d'
                                                , gallery_view
                                                , probe_view)
                    local gallery_feat = gallery_feat_tbl[gallery_view][gallery_j]
                    local probe_feat = probe_feat_tbl[probe_view][i]
                    local dst
                    
                    local dst = merge:forward(gallery_feat, probe_feat)
                    info(string.format('forawrd %s, %s, dst, %s, same: %s'
                                 , gallery_item, probe_item, dst, i == gallery_j))

                    table.insert(pro, 1 - dst)
                    table.insert(res, i == gallery_j)
                    if sim_mat[key_str] == nil then
                        sim_mat[key_str] = torch.zeros(n_persons, n_persons)
                    end
                    sim_mat[key_str][i][gallery_j] = dst
                    if i == gallery_j then
                        avg_same = avg_same  + dst
                        avg_same_count = avg_same_count + 1
                    else
                        avg_diff = avg_diff + dst
                        avg_dif_count = avg_dif_count + 1
                    end
                end
            end
            local eer = cal_eer(pro, res)
            local msg = string.format('for gallery view %s, probe view %s, equal error rate is %.06f', gallery_view, probe_view, eer)
            info(msg)
            table.insert(eer_str, msg)
        end
    end

    avg_same = avg_same / avg_same_count
    avg_diff = avg_diff / avg_dif_count
    for i=1,#eer_str do
        info(eer_str[i])
    end
    for _, gallery_view in ipairs(g_p_views) do
        for _, probe_view in ipairs(g_p_views) do
            local key_str = string.format('%02d-%02d', gallery_view, probe_view)
            local cmc = torch.zeros(n_persons)
            local sampling_order = torch.zeros(n_persons,n_persons)
            for i = 1,n_persons do
                local tmp = sim_mat[key_str][{i,{}}]
                local y,o = torch.sort(tmp)

                --find the element we want
                local indx = 0
                local tmp_idx = 1
                for j = 1,n_persons do
                    if o[j] == i then
                        indx = j
                    end

                    -- build the sampling order for the next epoch
                    -- we want to sample close images i.e. ones confused with this person
                    if o[j] ~= i then
                        sampling_order[i][tmp_idx] = o[j]
                        tmp_idx = tmp_idx + 1
                    end
                end

                for j = indx,n_persons do
                    cmc[j] = cmc[j] + 1
                end
            end
            cmc = (cmc / n_persons) * 100
            sum_precision = sum_precision + cmc[1]
            local cms_str = string.format('for gallery view:%02d, test view:%02d', gallery_view, probe_view)
            for c = 1,5 do
                if c <= n_persons then
                    cms_str = cms_str .. ' ' .. torch.floor(cmc[c] * 100) / 100.0
                end
            end
            info(cms_str)
        end
    end
    local avg_precision = sum_precision / ( #g_p_views * #g_p_views)
    model:training()
    return avg_same, avg_diff, avg_precision
end
