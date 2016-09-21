function X = sparsify(X)
%Squeeze out zero components in the sparse matrix
%Version 1: 10/26/2009
%Written by Mingyuan Zhou, Duke ECE, mz1@ee.duke.edu
[i,j,x] = find(X);
[m,n] = size(X);
X = sparse(i,j,x,m,n);
end