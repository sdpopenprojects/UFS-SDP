%function [feaSubsets,res_gs,res_aio, res_gs_ps] = fs_unsup_udfs_single_func(dataset, exp_settings, ~)
% function [feaSubsets,res_gs,res_aio, res_gs_ps] = fs_unsup_udfs_single_func(X, Y, exp_settings, ~)
function [feaSubsets] = fs_unsup_udfs_single_func(X, exp_settings, ~)
% run UDFS feature selection algorithm

%======================setup===========================
FeaNumCandi = exp_settings.FeaNumCandi;
nKmeans = exp_settings.nKmeans;
prefix_mdcs = [];
if isfield(exp_settings, 'prefix_mdcs')
    prefix_mdcs = exp_settings.prefix_mdcs;
end
%======================================================

% disp(['dataset:',dataset]);
% [X, Y] = extractXY(dataset);
[nSmp, nDim] = size(X);
%nClass = length(unique(Y));
nClass = 2;

%======================setup===========================
%gammaCandi = 10.^(-5:5);%正则化参数，这里选10^-3吧
%lamdaCandi = 10.^(-5:5);%不知道这是干嘛的？默认值1000
gammaCandi = 10.^(-5);
lamdaCandi = 10.^(-5);
knnCandi = 5;
paramCell = fs_unsup_udfs_build_param(knnCandi, gammaCandi, lamdaCandi);
%======================================================

t_start = clock;
disp('UDFS ...');
feaSubsets = cell(length(paramCell), 1);
parfor i1 = 1:length(paramCell)
    fprintf('UDFS parameter search %d out of %d...\n', i1, length(paramCell));
    param = paramCell{i1};
    L = LocalDisAna(X', param);
    A = X'*L*X;
    W = fs_unsup_udfs(A, nClass, param.gamma);
    [~, idx] = sort(sum(W.*W,2),'descend');
    feaSubsets{i1,1} = idx;
end
t_end = clock;
t1 = etime(t_end,t_start);
disp(['exe time: ',num2str(t1)]);
% 
% t_start = clock;
% disp('evaluation ...');
% res_aio = cell(length(paramCell), length(FeaNumCandi));
% for i2 = 1:length(FeaNumCandi)
%     m = FeaNumCandi(i2);
%     parfor i1 = 1:length(paramCell)
%         fprintf('UDFS parameter evaluation %d outof %d  ... %d out of %d...\n', i2, length(FeaNumCandi), i1, length(paramCell));
%         idx = feaSubsets{i1,1};
%         res_aio{i1, i2} = evalUnSupFS(X, Y, idx(1:m), struct('nKm', nKmeans));%这里改成逻辑回归的算法，把每一个评价指标存进cell里
%     end
% end
% [res_gs, res_gs_ps] = grid_search_fs(res_aio);%再用网格搜索搜出最好的
% res_gs.feaset = FeaNumCandi;
% t_end = clock;
% t2 = etime(t_end,t_start);
% disp(['exe time: ',num2str(t2)]);
% res_gs.time = t1;
% res_gs.time2 = t2;

%save(fullfile(prefix_mdcs, [dataset, '_best_result_UDFS.mat']),'FeaNumCandi','res_gs','res_aio', 'res_gs_ps');
end