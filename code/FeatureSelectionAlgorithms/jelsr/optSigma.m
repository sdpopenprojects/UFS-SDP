function sigma = optSigma(X)
    N = size(X,1);%返回矩阵的行数
    dist = EuDist2(X,X);
    dist = reshape(dist,1,N*N);
    sigma = median(dist);