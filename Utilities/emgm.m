function label= emgm(X, cls_num, nlsp)
% function [model,llh,label]= emgm(X, cls_num, nlsp)
% Perform EM algorithm for fitting the Gaussian mixture model.
%   X: d x n data matrix
%   init: k (1 x 1) or label (1 x n, 1<=label(i)<=k) or center (d x k)
% Written by Michael Chen (sth4nth@gmail.com).
% initialization
fprintf('EM for Gaussian mixture: running ... \n');
R = initialization(X,cls_num,nlsp);
[~,label(1,:)] = max(R,[],2);
R = R(:,unique(label));
tol = 1e-6;
maxiter = 10;
llh = -inf(1,maxiter);
converged = false;
t = 1;
while ~converged && t < maxiter
    t = t+1;
    model = maximization(X,R,nlsp);
    clear R;
    [R, llh(t)] = expectation(X,model,nlsp);
    % output
    fprintf('Iteration %d of %d, logL: %.2f\n',t,maxiter,llh(t));
    % output
    %     subplot(1,2,1);
    %     plot(llh(1:t),'o-'); drawnow;
    [~,label(:)] = max(R,[],2);
    u = unique(label);   % non-empty components
    if size(R,2) ~= size(u,2)
        R = R(:,u);   % remove empty components
    else
        converged = llh(t)-llh(t-1) < tol*abs(llh(t));
    end
end
label=label';
if converged
    fprintf('Converged in %d steps.\n',t-1);
else
    fprintf('Not converged in %d steps.\n',maxiter);
end

function R = initialization(X, init,nlsp)
%
index = 1:nlsp:size(X,2);
X = X(:,index);
[d,n] = size(X);
if isstruct(init)  % initialize with a model
    R  = expectation(X,init);
elseif length(init) == 1  % random initialization
    k = init;
    idx = randsample(n,k);
    m = X(:,idx);
    [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
    [u,~,label] = unique(label);
    while k ~= length(u)
        idx = randsample(n,k);
        m = X(:,idx);
        [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
        [u,~,label] = unique(label);
    end
    R = full(sparse(1:n,label,1,n,k,n));
elseif size(init,1) == 1 && size(init,2) == n  % initialize with labels
    label = init;
    k = max(label);
    R = full(sparse(1:n,label,1,n,k,n));
elseif size(init,1) == d  %initialize with only centers
    k = size(init,2);
    m = init;
    [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
    R = full(sparse(1:n,label,1,n,k,n));
else
    error('ERROR: init is not valid.');
end


function [R, llh] = expectation(X, model,nlsp)
%
means = model.means;
covs = model.covs;
w = model.mixweights;
n = size(X,2)/nlsp;
k = size(means,2);
logRho = zeros(n,k);
for i = 1:k
    TemplogRho = loggausspdf(X,means(:,i),covs(:,:,i));
    Temp = reshape(TemplogRho,[nlsp n]);
    logRho(:,i) = sum(Temp);
end
logRho = bsxfun(@plus,logRho,log(w));
T = logsumexp(logRho,2);
llh = sum(T)/n; % loglikelihood
logR = bsxfun(@minus,logRho,T);
R = exp(logR);


function model = maximization(X, R,nlsp)
%
[d,n] = size(X);
R = R(reshape(ones(nlsp,1)*(1:size(R,1)),size(R,1)*nlsp,1),:);
k = size(R,2);
nk = sum(R,1);
w = nk/n;
means = bsxfun(@times, X*R, 1./nk);
Sigma = zeros(d,d,k);
sqrtR = sqrt(R);
for i = 1:k
    Xo = bsxfun(@minus,X,means(:,i));
    Xo = bsxfun(@times,Xo,sqrtR(:,i)');
    Sigma(:,:,i) = Xo*Xo'/nk(i);
    Sigma(:,:,i) = Sigma(:,:,i)+eye(d)*(1e-6); % add a prior for numerical stability
end
model.dim = d;
model.nmodels = k;
model.mixweights = w;
model.means = means;
model.covs = Sigma;



function y = loggausspdf(X, mu, Sigma)
%
d = size(X,1);
X = bsxfun(@minus,X,mu);
%   [R,p] = CHOL(A), with two output arguments, never produces an
%   error message.  If A is positive definite, then p is 0 and R
%   is the same as above.   But if A is not positive definite, then
%   p is a positive integer.
[U,p]= chol(Sigma);
if p ~= 0
    error('ERROR: Sigma is not PD.');
end
Q = U'\X;
q = dot(Q,Q,1);  % quadratic term (M distance)
c = d*log(2*pi)+2*sum(log(diag(U)));   % normalization constant
y = -(c+q)/2;

function s = logsumexp(x, dim)
%
% Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
%   By default dim = 1 (columns).
% Written by Michael Chen (sth4nth@gmail.com).
if nargin == 1,
    % Determine which dimension sum will use
    dim = find(size(x)~=1,1);
    if isempty(dim), dim = 1; end
end
% subtract the largest in each column
y = max(x,[],dim);
x = bsxfun(@minus,x,y);
s = y + log(sum(exp(x),dim));
i = find(~isfinite(y));
if ~isempty(i)
    s(i) = y(i);
end