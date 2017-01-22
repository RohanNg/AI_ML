function sigma2  = covarianceMatrix(X, mu)

[m n] = size(X);
sigma2 = zeros(n, n);
for i= 1:m
    meanNormalizedFeature = X(i,:)-mu';
    sigma2 = sigma2 + meanNormalizedFeature'*meanNormalizedFeature;
end;
    sigma2 = sigma2/m;