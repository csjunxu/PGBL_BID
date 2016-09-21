function phi = Samplephi(X_k,c0,d0,Yflag)
%Sample phi
%Version 1: 09/12/2009
%Version 2: 10/26/2009
%Written by Mingyuan Zhou, Duke ECE, mz1@ee.duke.edu
if nargin<4
    sumYflag = numel(X_k);
else
    sumYflag = nnz(Yflag);
end
ci = c0 + 0.5*sumYflag;
di = d0 + 0.5*sum(sum((X_k).^2));
phi = gamrnd(ci,1./di);
end