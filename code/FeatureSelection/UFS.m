function [train_data_sub, test_data_sub, Uidx] = UFS(train_dataset_path, UFS_NAME)
%16
load(train_dataset_path);

[Smp_num,Fea_num] = size(train_data);
FeaNumCandi = ceil(Fea_num*0.15)

train_data = double(train_data);
test_data = double(test_data);

switch UFS_NAME

case 'UFSoL'
    [feaSubsets] = rank_fir_ordinal_locality(test_data);
    %whos feaSubsets
    fea_idx = feaSubsets(1:FeaNumCandi);
    Uidx = fea_idx;
    test_data_sub = test_data(:, fea_idx);
    train_data_sub = train_data(:, fea_idx);
case 'UDFS'
    exp_settings = [];
    exp_settings.nKmeans = 20;
    exp_settings.prefix_mdcs = [];
    exp_settings.FeaNumCandi = ceil(Fea_num*0.15);
    [feaSubsets] = fs_unsup_udfs_single_func(test_data, exp_settings);
    fea_idx = cell(1, 1);
    fea_idx{1} = feaSubsets{1,1}(1:exp_settings.FeaNumCandi);
    Uidx = fea_idx{1,1};
    test_data_sub = test_data(:, fea_idx{1,1});
    train_data_sub = train_data(:, fea_idx{1,1});
case 'U2FS'
    options.simType = 'RBF';
    options.sigma = 'highdim';
    [feaSubsets] = u2fs(test_data, 2, FeaNumCandi, options);
    fea_idx = feaSubsets(1:FeaNumCandi);
    Uidx = fea_idx;
    test_data_sub = test_data(:, fea_idx);
    train_data_sub = train_data(:, fea_idx);
case 'SRCFS'
    [feaSubsets] = SRCFS(test_data);
    fea_idx = feaSubsets(1:FeaNumCandi);
    Uidx = fea_idx;
    test_data_sub = test_data(:, fea_idx);
    train_data_sub = train_data(:, fea_idx);
case 'SOGFS'
    [feaSubsets] = SOGFS(test_data);
    fea_idx = feaSubsets(1:FeaNumCandi);
    Uidx = fea_idx;
    test_data_sub = test_data(:, fea_idx);
    train_data_sub = train_data(:, fea_idx);
case 'SOCFS'
    [feaSubsets] = SOCFS(test_data);
    fea_idx = feaSubsets(1:FeaNumCandi);
    Uidx = fea_idx;
    test_data_sub = test_data(:, fea_idx);
    train_data_sub = train_data(:, fea_idx);
case 'RUFS'
    [feaSubsets] = Robust_unsupervised(test_data);
    fea_idx = feaSubsets(1:FeaNumCandi);
    Uidx = fea_idx;
    test_data_sub = test_data(:, fea_idx);
    train_data_sub = train_data(:, fea_idx);
case 'NDFS'
    [feaSubsets] = NDFS(test_data);
    fea_idx = feaSubsets(1:FeaNumCandi);
    Uidx = fea_idx;
    test_data_sub = test_data(:, fea_idx);
    train_data_sub = train_data(:, fea_idx);
case 'LLCFS'
    [feaSubsets] = LLCFS(test_data);
    fea_idx = feaSubsets(1:FeaNumCandi);
    Uidx = fea_idx;
    test_data_sub = test_data(:, fea_idx);
    train_data_sub = train_data(:, fea_idx);
case 'Inf_FS2020'
    [~, ~, fea_idx] = InfFS_U(test_data, 0.5);
    Uidx = fea_idx;
    test_data_sub = test_data(:, fea_idx);
    train_data_sub = train_data(:, fea_idx);
case 'Inf_FS'
    [feaSubsets] = infFS(test_data);
    fea_idx = feaSubsets(1:FeaNumCandi);
    Uidx = fea_idx;
    test_data_sub = test_data(:, fea_idx);
    train_data_sub = train_data(:, fea_idx);
case 'FSASL'
    [feaSubsets] = rank_fir_adaptive_structure_learning(test_data);
    fea_idx = feaSubsets(1:FeaNumCandi);
    Uidx = fea_idx;
    test_data_sub = test_data(:, fea_idx);
    train_data_sub = train_data(:, fea_idx);
case 'FMIUFS'
    [feaSubsets] = ufs_FMI(test_data);
    fea_idx = feaSubsets(1:FeaNumCandi);
    Uidx = fea_idx;
    test_data_sub = test_data(:, fea_idx);
    train_data_sub = train_data(:, fea_idx);
case 'CNAFS'
    [~,~,feaSubsets,~] = CNAFS(test_data', 2, 0.01, 10, 10, 0.1, 0.001, 3, 5);
    fea_idx = feaSubsets(1:FeaNumCandi);
    Uidx = fea_idx;
    test_data_sub = test_data(:, fea_idx);
    train_data_sub = train_data(:, fea_idx);
case 'jelsr_lle'
    exp_settings = [];
    exp_settings.nKmeans = 20;
    exp_settings.prefix_mdcs = [];
    exp_settings.FeaNumCandi = ceil(Fea_num*0.15);
    [feaSubsets] = fs_unsup_jelsr_lle_single_func(test_data, exp_settings);
    fea_idx = cell(1, 1);
    fea_idx{1} = feaSubsets{1,1}(1:exp_settings.FeaNumCandi);
    Uidx = fea_idx{1,1};
    test_data_sub = test_data(:, fea_idx{1,1});
    train_data_sub = train_data(:, fea_idx{1,1});
case 'GLSPFS'
    exp_settings = [];
    exp_settings.nKmeans = 20;
    exp_settings.prefix_mdcs = [];
    exp_settings.FeaNumCandi = ceil(Fea_num*0.15);
    [feaSubsets] = fs_unsup_glspfs_single_func(test_data, exp_settings);
    fea_idx = cell(1, 1);
    fea_idx{1} = feaSubsets{1,1}(1:exp_settings.FeaNumCandi);
    Uidx = fea_idx{1,1};
    test_data_sub = test_data(:, fea_idx{1,1});
    train_data_sub = train_data(:, fea_idx{1,1});
end