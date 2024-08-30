function [W_compute, Y, obj] = fs_unsup_jelsr(data, W_ori, ReducedDim,alpha,beta)

%%%%%%%% Input: data: nSmp*nFea;
%%%             W_ori: The original local similarity matrix
%%%             ReducedDim: the dimensionality for low dimensionality
%%%                         embedding $Y$
%%%             alpha and beta ar two parameters

[nSmp,nFea] = size(data);

%%%%%%%%%%%%%%%%%%% Normalization of W_ori
D_mhalf = full(sum(W_ori,2).^-.5); 
W = compute_W(W_ori,data,D_mhalf);%对权重矩阵 W 进行归一化和对称化
%W(1:5,:)
%size(W)

%%%%%%%%%%%%%%%%%% Eigen_decomposition
%[r,c,v] = find(isinf(data))%data中并没有nan、inf的存在(不是data的问题)
W(find(isnan(W))) = 0;
W(find(isinf(W))) = 0;
%find(isinf(W))
%disp('Y')
Y = compute_Y(data,W, ReducedDim, D_mhalf);%根据数据矩阵和权重矩阵W（表示样本之间的相似性关系），计算降维后的数据 Y。
% 看看Ay是不是还是都是NaN（OK了，改完后Ay变成正常数字了）   
Y(find(isnan(Y))) = 0;
Y(find(isinf(Y))) = 0;
if issparse(data)
    data = [data ones(size(data,1),1)];
    [nSmp,nFea] = size(data);
else
    sampleMean = mean(data);
    data = (data - repmat(sampleMean,nSmp,1));
end

%%% To minimize squared loss with L21 normalization
%%%%%%%%%%%% Initialization
AA = data'*data;
%disp('Ay')
Ay = data'*Y;%全是NaN （改完后Ay变成正常数字了）
%disp('W_compute')
W_compute = (AA+alpha*eye(nFea))\Ay;%全是NaN（改完后W_compute变成正常数字了）
%disp('d')
d = sqrt(sum(W_compute.*W_compute,2));%全是NaN（改完后d变成正常数字了）

itermax = 20;
obj = zeros(itermax,1);
feaK = data'*data; % modified by liang du
for iter = 1:itermax 
   %%%%%%%%%%%%%%%%%%% Fix D to updata W_compute, Y
   %disp('D')
   D = 2*spdiags(d,0,nFea,nFea);%(正常的)
   %%%%%%%%%%%%%%%% To updata Y
   A = (D*feaK+alpha*eye(nFea));  
   Temp  = A\(D*data'); 
   Temp =  data*Temp;
   Temp = W_ori-beta*eye(nSmp)+beta*Temp; 
   %%%%% Normalization
   Temp = compute_W(Temp,data,D_mhalf);
   %Temp(1:5,:)
   %%%%% Eigen_decomposition  
   Temp(find(isnan(Temp))) = 0;
   Temp(find(isinf(Temp))) = 0;
   %disp('Y')
   Y = compute_Y(data,Temp, ReducedDim, D_mhalf);%里面存在NAN值，
   Y(find(isnan(Y))) = 0;
   Y(find(isinf(Y))) = 0;
   %Y

   %%%%%%%%%%%%%%%%% To updata W
   B = D*data'*Y; 
   W_compute = A\B;
   
   %%%%%%%%%%%%%%%%%% Fix W and update D
   d = sqrt(sum(W_compute.*W_compute,2));
   
end 
end 
 