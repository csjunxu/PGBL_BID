%------------------------------------------------------------------
% Matlab code for Gaussian noise Denoising
% Please cite the following paper if you use this code:
%------------------------------------------------------------------
% Patch Group based Bayesian Learning for Blind Image Denoising 
% Jun Xu, Dongwei Ren, Lei Zhang, David Zhang 
% NTIRE: New Trends in Image Restoration and Enhancement, 
% workshop at ACCV 2016, Taipei, Taiwan. 
%------------------------------------------------------------------
% Copyright @ Jun Xu, Email: csjunxu@comp.polyu.edu.hk
%------------------------------------------------------------------
clear;
Original_image_dir  =    'images/';
fpath = fullfile(Original_image_dir, 'house.png');
im_dir  = dir(fpath);
im_num = length(im_dir);
% set parameters
c0 = 1e-6;
d0 = 1e-6;
e0 = 1e-6;
f0  = 1e-6;
Hyper.c0=c0;
Hyper.d0=d0;
Hyper.e0=e0;
Hyper.f0=f0;
Hyper.PatchSize = 8;
Hyper.step = 3;
nlsp = 6;
Hyper.nlsp = nlsp;
Hyper.MaxIteration = 20;

nSig = 40; % noise stand deviation

imPSNR = [];
imSSIM = [];
for i = 1:im_num
    %% read clean image
    IMname = regexp(im_dir(i).name, '\.', 'split');
    IMname = IMname{1};
    IMin0=im2double(imread(fullfile(Original_image_dir, im_dir(i).name)));
    %% generate noisy image
    randn('seed',0); 
    IMin = IMin0 + nSig/255*randn(size(IMin0));
    %% denoising
    [Iout,NoiseVar,~] = PGBL_BID(IMin,IMin0,Hyper);
    Iout(Iout>1)=1;
    Iout(Iout<0)=0;
    %% output
    imPSNR{Sample} = [imPSNR{Sample} csnr( Iout*255,IMin0*255, 0, 0 )];
    imSSIM{Sample}  = [imSSIM{Sample} cal_ssim( Iout*255, IMin0*255, 0, 0 )];
    imwrite(Iout, ['PGBL_BID_Gau_' IMname '_' num2str(nSig) '.png']);
    fprintf('%s : PSNR = %2.4f, SSIM = %2.4f \n',im_dir(i).name,csnr( Iout*255, IMin0*255, 0, 0 ),cal_ssim( Iout*255, IMin0*255, 0, 0 ));
end
mPSNR(Ran) = mean(imPSNR);
mSSIM(Ran) = mean(imPSNR);
result = sprintf('PGBL_BID_Gau_%d.mat',nSig);
save(result,'nSig','mPSNR','mSSIM','imPSNR','imSSIM');

