%% SQUARE EXAMPLE 2 seismic

% close all
% clear all
% clc

%% PARAMETERS
setup.n         = 100;
% setup.level     = 0.01;
setup.N_noise   = numel(setup.level);

%problem parameters
setup.problem   = 2;
setup.image     = 8;
setup.type      = 2;
setup.par1      = 1;
setup.par2      = 2*setup.n;
setup.par_name  = 'level, receiver';

%% GENERATE DATA
[A, b, x1, x0, err_x0, dev0, prob_info, err_x, k0, par0, xxOR, U, sigA, V] ...
                = generate_data_IRtool_iko(setup.n, setup.problem, setup.image, setup.level, 1, 1, 'type', setup.type, 'level', setup.par1, 'p', setup.par2);
setup.size      = size(A);
[n,d]           = size(A);

%% some parameter
ttt             = clock;
m               = [2*k0, 5*k0];
gkl             = k0+1000;
hy_iter         = 300;
xtol            = [1e-3 0];
ptol            = [0 0];
maxit           = [25 25];
k               = 1;
%% 1 ORACLE LS
xx(:,k)         = xxOR{1};
k               = k+1;
xx(:,k)         = xxOR{2};
err(k)          = err_x(xx(:,k));
k               = k+1;
%% 2 LS GCV
xx(:,k)    = LS_gcv_iko(U, sigA,V,b, x1);
err(k)     = err_x(xx(:,k));
k          = k+1;

%% 3 LS GCV eff
xx(:,k)          = LS_gcv_iko(eye(gkl), sigA(1:gkl),V(:, 1:gkl), U(:,1:gkl)'*b, x1);
err(k)           = err_x(xx(:,k));
k                = k+1;

%% 4-5 HYBRID
options          = HyBRset_iko('RegPar', 'WGCV', 'Iter', hy_iter, 'Omega', 'adapt', 'Reorth', 'on', 'x_true', xxOR{1});
[xx(:,k+1), out] = HyBR_iko(A, b, [], options);
err(k)           = out.Enrm(out.iterations);
err(k+1)         = out.Enrm(end);
options          = HyBRset_iko('RegPar', 'WGCV', 'Iter', out.iterations, 'Omega', 'adapt', 'Reorth', 'on', 'x_true', xxOR{1});
xx(:,k)          = HyBR_iko(A, b, [], options);
k                = k+2;
%% 6 HYBRID corrected ONE-SHOT
xx(:,k)          =  hybrid_corrected_one_shot_iko(A,b,gkl);
err(k)           = err_x(xx(:,k));
k                = k+1;

%% 7 PD M-IHS GKL
xx(:,k)          = reg_pd_mihs_gkl_lower_iko(A,b,m,x1,xtol,ptol,maxit);
err(k)           = err_x(xx(:,k));
k                = k+1;

%% SAVE
save(sprintf('image_02_sq_d%d%d_t%d%d', ttt(3), ttt(2), ttt(4), ttt(5)), 'setup', 'prob_info','b','x0','err','xx')



