%% OVER-DETERMIEND EXAMPLE 2 seismic

% close all
% clear all
% clc

%% PARAMETERS
n               = 1500;
d               = 40000;
% setup.level     = 0.01;
setup.N_noise   = numel(setup.level);

%problem parameters
setup.profile = 10;
setup.kappa   = 8;
setup.max     = 100;
setup.matrix_dev = 6;
setup.matrix_cor = 0.9;

%% GENERATE DATA
[A, b, x0, x1, dev0, U, sigA, V] = generate_data(n,d,...
    'correlated',               true,...
    'matrix correlation',       setup.matrix_cor,...
    'matrix deviation',         setup.matrix_dev,...
    'hansen singular values',   setup.profile,...
    'hansen max',               setup.max,...
    'kappa',                    setup.kappa,...
    'noise level',              setup.level,...
    'hansen input',             false,...
    'input dev',                1,...
    'prior dev',                0,...
    'prior mean',               0,...
    'structure',                false,...
    'plot',                     false);
setup.size      = size(A);

%% ORACLE Regularized SOL
[xxOR, parOR, errOR]   = oracle_methods(U, sigA, V,b,x0,x1);
err_x           = @(xx)(errOR{1}(xx,1));
k0              = parOR(1);
setup.k0        = k0;
%% some parameter
ttt             = clock;
m               = 2*n;
gkl             = k0+300;
hy_iter         = 300;
xtol            = 1e-3;
ptol            = 0;
maxit           = 30;
k               = 1;
%% 1 ORACLE LS
xx(:,k)         = xxOR{1};
k               = k+1;
xx(:,k)         = xxOR{2};
err(k)          = err_x(xx(:,k));
k               = k+1;
%% 2 LS GCV
xx(:,k)         = LS_gcv_iko(U, sigA,V,b, x1);
err(k)          = err_x(xx(:,k));
k               = k+1;

%% 4-5 HYBRID
options          = HyBRset_iko('RegPar', 'WGCV', 'Iter', hy_iter, 'Omega', 'adapt', 'Reorth', 'on', 'x_true', xxOR{1});
[xx(:,k+1), out] = HyBR_iko(A, b, [], options);
err(k)           = out.Enrm(out.iterations);
err(k+1)         = out.Enrm(end);
options          = HyBRset_iko('RegPar', 'WGCV', 'Iter', out.iterations, 'Omega', 'adapt', 'Reorth', 'on', 'x_true', xxOR{1});
xx(:,k)          = HyBR_iko(A, b, [], options);
k                = k+2;

%% 5 HYBRID corrected ONE-SHOT
xx(:,k)          =  hybrid_corrected_one_shot_iko(A,b,gkl);
err(k)           = err_x(xx(:,k));
k                = k+1;

%% 6 PD M-IHS GKL
xx(:,k)          = reg_dual_mihs_gkl_iko(A,b,m,x1,xtol,ptol,maxit);
err(k)           = err_x(xx(:,k));
k                = k+1;

%% SAVE
save(sprintf('image_05_under_d%d%d_t%d%d', ttt(3), ttt(2), ttt(4), ttt(5)), 'setup','b','x0','err','xx')
