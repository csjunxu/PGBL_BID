function [D,S,Z,phi,alpha,Pi] = InitMatrix_Denoise(X_k,K,InitOption,IsSeparateAlpha,phi)
[P,N]=size(X_k);
if nargin<4
    IsSeparateAlpha = false;
end
if IsSeparateAlpha == false
    alpha = 1;
else
    alpha = ones(1,K);
end

if strcmp(InitOption,'SVD')==1
    [U_1,S_1,V_1] = svd(full(X_k),'econ');
    if P<=K
        D = zeros(P,K);
        D(:,1:size(S_1,2)) = U_1*S_1; % D(:,1:P) = U_1*S_1;
        S = zeros(N,K);
        S(:,1:size(V_1,2)) = V_1; %S(:,1:P) = V_1;
    else
        D =  U_1*S_1;
        D = D(1:P,1:K);
        S = V_1;
        S = S(1:N,1:K);
    end
    Z = true(N,K);
    Pi = 0.5*ones(1,K);
else
    %     if size(IMin,3)==1
    %% Setting 1
    D = randn(P,K)/sqrt(P);
    S = randn(N,K);
    Z = logical(sparse(N,K));
    Pi = 0.01*ones(1,K);
    D(:,1) = mean(X_k,2)*100;
    S(:,1)=1/100;
    Z(:,1)= true;
    %% Setting 2
    % D = randn(P,K)/sqrt(P);
    % S = sparse(zeros(N,K));
    % Z = logical(sparse(N,K));
    % Pi = 1e-6*ones(1,K);
    % D(:,1) = mean(X_k,2)*100;
    % S(:,1) = 1;
    % Z(:,1)= true;
    % Pi(1) = 0.9;
    %     end
end
S = S.*Z;
end





