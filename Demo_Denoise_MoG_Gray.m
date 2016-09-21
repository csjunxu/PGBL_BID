%------------------------------------------------------------------
% Matlab code for Moxture of Gaussian noise Denoising
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
Hyper.step = 2;
nlsp = 6;
Hyper.nlsp = nlsp;
Hyper.MaxIteration = 20;
% MoG noise stand deviation
nSig = [10 30 100]; % [5 15 25];
% weights of different noise levels
nWeight = [0.25 0.5 0.25];

imPSNR = [];
imSSIM = [];
for i = 1:im_num
    %% read clean image
    IMname = regexp(im_dir(i).name, '\.', 'split');
    IMname = IMname{1};
    IMin0=im2double(imread(fullfile(Original_image_dir, im_dir(i).name)));
    %% generate MoG noise
    randn('seed',0)
    SampleIndex = randperm(numel(IMin0));
    NoiseMatrix = zeros(size(IMin0));
    randn('seed',0)
    Pixels1 = fix(nWeight(1)*numel(NoiseMatrix));
    NoiseMatrix(SampleIndex(1 : Pixels1)) = nSig(1)/255*randn(1,Pixels1);
    randn('seed',0)
    Pixels2 = fix(nWeight(2)*numel(NoiseMatrix));
    NoiseMatrix(SampleIndex(Pixels1+1 : Pixels1+Pixels2)) = nSig(2)/255*randn(1,Pixels2);
    randn('seed',0)
    Pixels3 = numel(NoiseMatrix) - (Pixels1+Pixels2);
    NoiseMatrix(SampleIndex(Pixels1+Pixels2+1 : end)) = nSig(3)/255*randn(1,Pixels3);
    %% generate noisy image with MoG noise
    IMin = IMin0 + NoiseMatrix;
    fprintf('%s :\n',im_dir(i).name);
    imwrite(IMin, ['Noisy_MoG_' IMname '_' num2str(nSig(1)) '_' num2str(nWeight(1)) '_' num2str(nSig(2)) '_' num2str(nWeight(2)) '_' num2str(nSig(3)) '_' num2str(nWeight(3)) '.png']);
    %% denoising
    [Iout,NoiseVar,~] = PGBL_BID(IMin,IMin0,Hyper);
    Iout(Iout>1)=1;
    Iout(Iout<0)=0;
    %% output
    imPSNR = [imPSNR csnr( Iout*255,IMin0*255, 0, 0 )];
    imSSIM  = [imSSIM cal_ssim( Iout*255, IMin0*255, 0, 0 )];
    imwrite(Iout, ['PGBL_BID_MoG_' IMname '_' num2str(nSig(1)) '_' num2str(nWeight(1)) '_' num2str(nSig(2)) '_' num2str(nWeight(2)) '_' num2str(nSig(3)) '_' num2str(nWeight(3)) '.png']);
    fprintf('%s : PSNR = %2.4f, SSIM = %2.4f \n',im_dir(i).name,csnr( Iout*255, IMin0*255, 0, 0 ),cal_ssim( Iout*255, IMin0*255, 0, 0 ));
end
mPSNR = mean(imPSNR);
mSSIM = mean(imSSIM);
result = sprintf('PGBL_BID_MoG_%d_%2.2f_%d_%2.2f_%d_%2.2f.mat',nSig(1),nWeight(1),nSig(2),nWeight(2),nSig(3),nWeight(3));
save(result,'nSig','imPSNR','imSSIM','mPSNR','mSSIM');