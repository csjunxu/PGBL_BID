%------------------------------------------------------------------
% Matlab code for mixed Gaussion and spiky noise Denoising
% Please cite the following paper if you use this code:
%------------------------------------------------------------------
% Patch Group based Bayesian Learning for Blind Image Denoising 
% Jun Xu, Dongwei Ren, Lei Zhang, David Zhang 
% NTIRE: New Trends in Image Restoration and Enhancement, 
% workshop at ACCV 2016, Taipei, Taiwan. 
%------------------------------------------------------------------
% Copyright @ Jun Xu, Email: csjunxu@comp.polyu.edu.hk
% The code can only used in non-commercial usage.
%------------------------------------------------------------------

File list:

Demo files:
    Demo_Denoise_Real
    Demo_Denoise_Gaussian_Gray.m
    Demo_Denoise_GauSpi_Gray.m
    Demo_Denoise_MoG_Gray.m
    Demo_Denoise_GauLocalVar_Gray.m
    Demo_Denoise_PoiGau_Gray_VST.m 
    GenAnscombe_vectors.mat: used in Demo_Denoise_PoiGau_Gray_VST.m 

Utilities/

Main programs:
PGBL_BID.m: The Patch Group based Bayesian Learning for Blind Image Denoising program.

Subprograms for Gibbs sampling:
SampleDZS.m: Sampling the dictionary D, the binary indicating matrix Z, and the pseudo weight matrix S. Used for no missing data case.
SamplePi.m: Sampling Pi.
Samplephi.m: Sampling the noise precision phi.
Samplealpha: Sampling alpha, the precision of si.

Subprograms for the updates in sequential learning:
SZUpdate.m: Update the pseudo weight matrix S and binary indicating matrix Z in sequential learning.

Other subprograms:
sparsity.m: Squeeze out zero components in the sparse matrix.
InitMatrix_Denoise.m: Initialization for gray-scale image denoising

images/

house.png: the original house image. 
Real_Niaochaogirls.png: the original real noisy image, cropped from Neat Image Website.
Other test images can be downloaded from:
1: http://www.cs.tut.fi/~foi/GCF-BM3D/
2: http://www4.comp.polyu.edu.hk/~csjunxu/Publications.html
3: https://ni.neatvideo.com/examples#bird
4: http://www.ipol.im/pub/art/2015/125/
