%------------------------------------------------------------------
% Matlab code for mixed Possion and Gaussion noise Denoising
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
addpath('./PoissonFunction/');

peaks = [60 30 10];  % target peak values for the scaled image
sigmas = peaks/10;                % standard deviation of the Gaussian noise
reps = 1;                        % number of replications (noise realizations)
for pp=1:numel(peaks)
    randn('seed',0);    % fixes seed of random noise
    rand('seed',0);
    % mixed Poisson-Gaussian noise parameters:
    peak = peaks(pp); % target peak value for the scaled image
    % Poisson scaling factor
    alpha = 1;
    % Gaussian component N(g,sigma^2)
    sigma = sigmas(pp);
    g = 0.0;
    for Sample =1:reps
        PSNR_input = zeros(1,im_num);
        PSNR_yhat = zeros(1,im_num);
        PSNR_yhat_cfa = zeros(1,im_num);
        PSNR_yhat_asy = zeros(1,im_num);
        PSNR_yhat_alg = zeros(1,im_num);
        for i = 1:im_num
            %% read clean image
            IMname = regexp(im_dir(i).name, '\.', 'split');
            IMname = IMname{1};
            IMin0=im2double(imread(fullfile(Original_image_dir, im_dir(i).name)));
            scaling = peak/max(IMin0(:));
            sIMin0 = scaling*IMin0;
            %% Generate noisy observation
            IMin = alpha*poissrnd(sIMin0) + sigma*randn(size(sIMin0)) + g;
            PSNR_input(i) = 10*log10(peak^2/(mean((sIMin0(:)-IMin(:)).^2)));
            disp(['Peak = ' num2str(peak) ', sigma = ' num2str(sigma)])
            disp(['input PSNR = ' num2str(PSNR_input(i))])
            fprintf('%s :\n',im_dir(i).name);
            imwrite(IMin/peak, ['Noisy_PoiGau_' IMname '_' num2str(peak) '_' num2str(alpha) '.png']);
            %% Apply forward variance stabilizing transformation
            % Generalized Anscombe VST (J.L. Starck, F. Murtagh, and A. Bijaoui, Image  Processing  and  Data Analysis, Cambridge University Press, Cambridge, 1998)
            fz = GenAnscombe_forward(IMin,sigma,alpha,g);
            %% DENOISING
            sigma_den = 1;  % Standard-deviation value assumed after variance-stabiliation
            % Scale the image (BM3D processes inputs in [0,1] range)
            scale_range = 1;
            scale_shift = (1-scale_range)/2;
            
            maxzans = max(fz(:));
            minzans = min(fz(:));
            fz = (fz-minzans)/(maxzans-minzans);
            sigma_den = sigma_den/(maxzans-minzans);
            fz = fz*scale_range+scale_shift;
            sigma_den = sigma_den*scale_range;
            
            %% denoising assuming AWGN
            [Iout,NoiseVar,~] = PGBL_BID(fz,sIMin0/peak,Hyper);
            Iout(Iout>1)=1;
            Iout(Iout<0)=0;
            % Scale back to the initial VST range
            Iout = (Iout-scale_shift)/scale_range;
            Iout = Iout*(maxzans-minzans)+minzans;
            
            %% Apply the inverse transformation
            sIMin0hat = GenAnscombe_inverse_exact_unbiased(Iout,sigma,alpha,g);   % exact unbiased inverse
            sIMin0hat_cfa = GenAnscombe_inverse_closed_form(Iout,sigma,alpha,g);  % closed-form approximation
            sIMin0hat_asy =  (Iout/2).^2 - 1/8 - sigma^2;                       % asymptotical inverse
            sIMin0hat_alg =  (Iout/2).^2 - 3/8 - sigma^2;                       % algebraic inverse
            
            PSNR_sIMin0hat(i)   =   10*log10(peak^2/mean((sIMin0(:)-sIMin0hat(:)).^2));
            SSIM_sIMin0hat(i)   =   cal_ssim(sIMin0,sIMin0hat,0,0);
            PSNR_sIMin0hat_cfa(i) = 10*log10(peak^2/mean((sIMin0(:)-sIMin0hat_cfa(:)).^2));
            PSNR_sIMin0hat_asy(i) = 10*log10(peak^2/mean((sIMin0(:)-sIMin0hat_asy(:)).^2));
            PSNR_sIMin0hat_alg(i) = 10*log10(peak^2/mean((sIMin0(:)-sIMin0hat_alg(:)).^2));
            %% output
            imwrite(sIMin0hat/peak, ['PGBL_BID_PoiGau_' IMname '_peak' num2str(peak) '_alpha' num2str(alpha) '.png']);
            fprintf('%s : PSNR = %2.4f, SSIM = %2.4f \n',im_dir(i).name,PSNR_sIMin0hat(i),SSIM_sIMin0hat(i));
        end
        disp(' ')
        disp(['Avg output PSNR (exact unbiased inv.)  = ' num2str(mean(PSNR_sIMin0hat))])
        disp(['Avg output PSNR (closed-form approx.)  = ' num2str(mean(PSNR_sIMin0hat_cfa))])
        disp(['Avg output PSNR (asymptotical inverse) = ' num2str(mean(PSNR_sIMin0hat_asy))])
        disp(['Avg output PSNR (algebraic inverse)    = ' num2str(mean(PSNR_sIMin0hat_alg))])
        disp(' ')
        fprintf('The average PSNR = %2.4f, SSIM = %2.4f. \n', num2str(mean(PSNR_sIMin0hat)),num2str(mean(SSIM_sIMin0hat)));
        result = sprintf('PGBL_BID_PoiGau_peak%d_alpha%d_sigma%d_Sample%d.mat',peak,alpha,sigma,Sample);
        save(result,'PSNR_sIMin0hat','SSIM_sIMin0hat','PSNR_sIMin0hat_cfa','PSNR_sIMin0hat_asy','PSNR_sIMin0hat_alg');
    end
end