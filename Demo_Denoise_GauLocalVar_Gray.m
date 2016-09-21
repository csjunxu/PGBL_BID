%------------------------------------------------------------------
% Matlab code for GauLocalVar noise Denoising
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
Hyper.MaxIteration = 20;
nlsp = 6;
Hyper.nlsp = nlsp;

scale = 0.03;
imPSNR = [];
imSSIM = [];
for i = 1:im_num
    %% read clean image
    IMname = regexp(im_dir(i).name, '\.', 'split');
    IMname = IMname{1};
    IMin0=im2double(imread(fullfile(Original_image_dir, im_dir(i).name)));
    rand('seed',0);
    V = scale*rand(size(IMin0));
    %% Generate noisy observation
    IMin = imnoise(IMin0,'localvar',V);
    fprintf('%s :\n',im_dir(i).name);
    imwrite(IMin, ['Noisy_GauLocVal_' IMname '_' num2str(scale) '.png']);
    %% denoising
    [Iout,NoiseVar,~] = PGBL_BID(IMin,IMin0,Hyper);
    Iout(Iout>1)=1;
    Iout(Iout<0)=0;
    imPSNR = [imPSNR csnr( Iout*255,IMin0*255, 0, 0 )];
    imSSIM  = [imSSIM cal_ssim( Iout*255, IMin0*255, 0, 0 )];
    %% output
    imwrite(Iout, ['PGBL_BID_GauLocVal_' IMname '_' num2str(scale) '.png']);
    fprintf('%s : PSNR = %2.4f, SSIM = %2.4f \n',im_dir(i).name,csnr( Iout*255, IMin0*255, 0, 0 ),cal_ssim( Iout*255, IMin0*255, 0, 0 ));
end
mPSNR = mean(imPSNR);
mSSIM = mean(imSSIM);
fprintf('The average PSNR = %2.4f, SSIM = %2.4f. \n', mPSNR,mSSIM);
result = sprintf('PGBL_BID_GauLocVal_scale%2.2.mat',scale);
save(result,'mPSNR','mSSIM','imPSNR','imSSIM');
