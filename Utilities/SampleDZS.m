function [X_k, D, Z, S] = SampleDZS(X_k, D, Z, S, Pi, alpha, phi, Dsample, Zsample, Ssample)
%Sample the Dictionary D, the Sparsity Pattern Z, and the Pesudo Weights S
%when there are no missing data;

%Input:
%X_k: the residual error, X_k = X - D*(Z.*S)';
%D: the dictionary (factor loading);
%Z: the binary indicator matrix (factor score sparsity pattern);
%S: the pesudo weight matrix (factor score);
%Pi: the parameter of the bernoulli process;
%alpha: the precision for S, a single precision is used if
%length(alpha)==1, otherwise, a separate precision for each factor score
%vector is used;
%phi: the noise precision;
%Dsample: sample D if it is TRUE;
%Zsample: sample Z if it is TRUE;
%Ssample: sample S if it is TRUE;

%Output:
%Updated X_k, D, Z, S;

%Note that D, S and Z can be sampled in different ways than the one coded
%here. S can be sampled row by row and Z can be sampled by collapsing S, as
%shown in the end. The inference equations can be found in "John Paisley,
%Mingyuan Zhou, Guillermo Sapiro and Lawrence Carin, Nonparametric image
%interpolation and dictioanry learing using spatially-dependent Dirichlet
%and Beta process Priors, submitted to ICIP 2010".

%Version 1: 10/21/2009;
%Version 2: 10/26/2009;
%Version 3: 10/28/2009;
%Updated in 03/08/2010;

%Written by Mingyuan Zhou, Duke ECE, mz1@ee.duke.edu,
%mingyuan.zhou@duke.edu

if nargin<8
    Dsample = true;
end
if nargin<9
    Zsample = true;
end
if nargin<10
    Ssample = true;
end

[P,N] = size(X_k);
K = size(D,2);

if length(alpha)==1
    alpha = repmat(alpha,1,K);
elseif length(alpha)==2
   alpha = [alpha(1),repmat(alpha(2),1,K-1)];
end

for k=1:K
    nnzk = nnz(Z(:,k));
    if nnzk>0
        X_k(:,Z(:,k)) = X_k(:,Z(:,k)) + D(:,k)*S(Z(:,k),k)';    
    end
    
    if Dsample
        %Sample D
        sig_Dk = 1./(phi*sum(S(Z(:,k),k).^2)+P);
        mu_D = phi*sig_Dk* (X_k(:,Z(:,k))*S(Z(:,k),k));
        D(:,k) = mu_D + randn(P,1)*sqrt(sig_Dk);        
    end
    
    if Zsample || Ssample
        DTD = sum(D(:,k).^2);
    end
    
    if Zsample
        %Sample Z        
        Sk = full(S(:,k));
        %draw the Pesudo Weights S(i,k) from prior if Z(i,k)=0
        Sk(~Z(:,k)) = randn(N-nnz(Z(:,k)),1)*sqrt(1/alpha(k));
        temp =  - 0.5*phi*( (Sk.^2 )*DTD - 2*Sk.*(X_k'*D(:,k)) );
        temp = exp(temp).*Pi(:,k);
        Z(:,k) = sparse( rand(N,1) > ((1-Pi(:,k))./(temp+1-Pi(:,k))) );
        %temp = temp + log(Pi(:,k)+realmin) - log(1-Pi(:,k)+realmin);
        %Z(:,k) = ( rand(N,1) > (1./(exp(temp)+1)) );

    end
    
    nnzk = nnz(Z(:,k));    
    if Ssample
        if nnzk>0
        %Sample S
        sigS1 = 1/(alpha(k) + phi*DTD);
        %S(Z(:,k),k) = randn(nnz(Z(:,k)),1)*sqrt(sigS1)+ sigS1*(phi*(X_k(:,Z(:,k))'*D(:,k)));
        %S(~Z(:,k),k) = randn(N-nnz(Z(:,k)),1)*sqrt(1/alpha);
        S(:,k) = sparse(find(Z(:,k)),1,randn(nnz(Z(:,k)),1)*sqrt(sigS1)+ sigS1*(phi*(X_k(:,Z(:,k))'*D(:,k))),N,1);        
        else
            S(:,k)  = 0;
        end
    end
    
    if nnzk>0
        X_k(:,Z(:,k)) = X_k(:,Z(:,k))- D(:,k)*S(Z(:,k),k)';
    end
end
Z = sparsify(Z);
S = sparsify(S);

% % Note: the codes below have note been fully tested.
% % Sample S row by row
% ST = S';
% ZT = Z';
% X_k = X_k + D*S';
% for i=1:N
%     D_tilde = D.*repmat(ZT(:,i)',P,1);
%     temp = chol(phi*D_tilde'*D_tilde + diag(alpha))\eye(K);
%     ST(:,i) = temp*(randn(K,1)+temp'*phi*D_tilde'*X_k(:,i));
% end
% ST = ST.*ZT;
% S = ST';
% X_k = X_k - D*S';
% 
% % Sample Z by collapsing S
% X_k = X_k + D*S';
% ST = S';
% ZT = Z';
% for k=1:K
%   S(~Z(:,k),k) = randn(N-nnz(Z(:,k)),1)*sqrt(1/alpha(k));
% end
% 
% for i=1:N
%     D_tilde = D(:,ZT(:,i)==1);
%     [P,Kt] = size(D_tilde);
%     if Kt>=P
%         MMinv = (1/phi*eye(P)+1/alpha*(D_tilde*D_tilde'))\eye(P);
%     else
%         MMinv = phi*eye(P) - phi^2*D_tilde*((diag(alpha) + phi*(D_tilde'*D_tilde))\eye(Kt))*D_tilde';
%     end
%     QZ1=zeros(K,1);
%     for k=1:K
%         if ZT(k,i)==0
%             Mik=MMinv;
%         else
%             Dk = D(:,k);
%             Mik = MMinv + (MMinv*Dk)*(Dk'*MMinv)/(alpha(k) - Dk'*MMinv*Dk);
%         end
%         DMD = D(:,k)'*Mik*D(:,k);
%         QZ1(k) = -0.5*log(1+1/alpha(k)*DMD ) + 0.5*(D(:,k)'*Mik*X_k(:,i))^2/(alpha(k)+AMA );
%     end
%     temp = exp(QZ1).*Pi;
%     ZT(:,i) = binornd(1,1-(1-Pi)./(temp+1-Pi));
% end
% Z = ZT';
% S = S.*Z; 
% X_k = X_k - D*S';

