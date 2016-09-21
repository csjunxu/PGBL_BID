function       [nDCSeedX,nDCnlX,blk_arr,DC] = CalNonLocal( im, ps, step,nlsp)
% record the non-local patch set and the index of each patch in
% of seed patches in image
[h,  w, ch]      =  size(im);
Win = 2*ps-1;
maxr = h-ps+1;
maxc = w-ps+1;
r          =  1:step:maxr;
r          =  [r r(end)+1:maxr];
c          =  1:step:maxc;
c          =  [c c(end)+1:maxc];
lenr = length(r);
lenc = length(c);
X          =  zeros(ps^2*ch, maxr*maxc, 'double');
l   =  0;
for k = 1:ch
    for i  = 1:ps
        for j  = 1:ps
            l   =  l+1;
            blk  = im(i:end-ps+i,j:end-ps+j,k);
            X(l,:) = blk(:)';
        end
    end
end
% index of each patch in image
Index     =   (1:maxr*maxc);
Index    =   reshape(Index,maxr,maxc);
% record the indexs of patches similar to the seed patch
blk_arr   =  zeros(nlsp*lenr*lenc, 1, 'double');
% non-local patch sets of X
DC = zeros(ps^2*ch,nlsp*lenr*lenc,'double');
nDCSeedX = zeros(ps^2*ch,lenr*lenc,'double');
nDCnlX = zeros(ps^2*ch,lenr*lenc*nlsp,'double');
for  i  =  1 :lenr
    for  j  =  1 : lenc
        row = r(i);
        col = c(j);
        off = (col-1)*maxr + row;
        off1 = (j-1)*lenr + i;
        % the range indexes of the window for searching the similar patches
        rmin    =   max( row-Win, 1 );
        rmax    =   min( row+Win, maxr );
        cmin    =   max( col-Win, 1 );
        cmax    =   min( col+Win, maxc );
        idx     =   Index(rmin:rmax, cmin:cmax);
        idx     =   idx(:);
        neighbor       =   X(:,idx); % the patches around the seed in X
        seed       =   X(:,off);
        dis = sum(bsxfun(@minus,neighbor, seed).^2,1);
        [~,ind]   =  sort(dis);
        indc        =  idx( ind( 1:nlsp ) );
        blk_arr((off1-1)*nlsp+1:off1*nlsp)  =  indc;
        temp = X( : , indc );
        DC(:,(off1-1)*nlsp+1:off1*nlsp) = repmat(mean(temp,2),[1 nlsp]);
        variations = temp - DC(:,(off1-1)*nlsp+1:off1*nlsp);
        nDCnlX(:,(off1-1)*nlsp+1:off1*nlsp) = variations;
        nDCSeedX(:,off1) = variations(:,1);
    end
end