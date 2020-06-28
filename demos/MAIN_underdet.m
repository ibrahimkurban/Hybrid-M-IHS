clear all
close all
clc

%% SPEC
%size
n       = 1e3;
d       = 14e3;

LEVEL       = 0.01; % noise level
KAPPA       = 7; % condition number
MAX         = 1; % largest singular value 
PROFILE     = 7; % you can change singualr value profil from here 1-10
HYBRID_iter = 300; % number of hybrid-lsqr iteration


%% GENERATE DATA
[A, b, x0, x1, dev, U, sigA, V] = generate_data(n,d,...
    'correlated',               true,...
    'matrix correlation',       0.9,...
    'matrix deviation',         5,...
    'hansen singular values',   PROFILE,...
    'hansen max',               MAX,...
    'kappa',                    KAPPA,...
    'noise level',              LEVEL,...
    'hansen input',             false,...
    'input dev',                1,...
    'prior dev',                0,...
    'prior mean',               0,...
    'structure',                false,...
    'plot',                     true);
fprintf('noise deviation: %2.2e\n', dev)


%% ORACLE Regularized SOL
[xxOR, parOR, errOR, err_rid] = oracle_methods(U, sigA, V,b,x0,x1);
err_x   = @(xx)(errOR{1}(xx,1));
% err_xreg= @(xx)(errOR{2}(xx,1));
k0      = parOR(1);
par0     = parOR(2);
fprintf('effective rank : %d\n', k0);
fprintf('Oracle reg. par: %1.2e\n', par0);
% xtrun       = V(:,1:k0)*((sigA(1:k0).^-1).*(U(:,1:k0)'*b));
% err_xtrun   = @(xx)(sqrt(sum((xtrun - xx).^2, 1))/norm(xtrun));

%% Full LS Soution
[x_gcv, par_gcv, dev_est1] = LS_gcv_iko(U, sigA,V,b, x1);

fprintf('Reg LS GCV reg. par    : %1.2e\n', par_gcv);


%% Hybrid
options         = HyBRset_iko('RegPar', 'WGCV', 'Iter', HYBRID_iter, 'Omega', 'adapt','Reorth', 'on', 'x_true', xxOR{1});
fprintf('\nHyBR runs...'); tic;
[x_hy, output] = HyBR_iko(A, b, [], options);
fprintf('%2.1e sec elapsed\n\n', toc);


%% methods
xtol        = 1e-3;
maxit       = 25;
ptol        = 0;
L           = k0+500; %normally L = min(n,d)
params.L    = L;
m           = 2*n;

[~, xx1, pari1, in_iter1]       = reg_dual_mihs_svd_iko(A,b,m,x1,xtol,ptol,maxit,params);
[~, xx2, pari2, in_iter2]       = reg_dual_mihs_gkl_iko(A,b,m,x1,xtol,ptol,maxit,params);

%% plot
figure;
hold on; grid on;
xlabel('iterate, k'); ylabel('oracle error');
plot(log10(err_rid)*ones(HYBRID_iter,1), 'k:', 'linewidth', 2);
plot(log10(err_x(x_gcv))*ones(HYBRID_iter,1), 'r:', 'linewidth', 2);
plot(log10(output.Enrm(1:end)), '-.', 'linewidth', 2);
plot(output.iterations, log10(output.Enrm(output.iterations)), 'o', 'markersize', 10)
plot(log10(err_x(xx1)), 'linewidth', 2);
plot(log10(err_x(xx2)), 'linewidth', 2);
% plot(log10(err_x(xx3)), 'x-', 'linewidth', 2);
% plot(log10(err_x(xx4)), 'o-', 'linewidth', 2);
legend('Oracle Reg', 'LS GCV',  'Hybrid wgcv', 'GCV stop', 'Dual MIHS (svd)', 'Dual MIHS (gkl)')
figure;
hold on; grid on;
xlabel('iterate, k'); ylabel('\lambda');
plot(log10(parOR(2))*ones(5,1), 'k:', 'linewidth', 2);
plot(log10(par_gcv)*ones(5,1), 'r:', 'linewidth', 2);
plot(log10(output.Alpha)*2, '-.', 'linewidth', 2)
plot(output.iterations, 2*log10(output.Alpha(output.iterations)), 'o', 'markersize', 10)
plot(log10(pari1), 'linewidth', 2);
plot(log10(pari2), 'linewidth', 2);
legend('Oracle Reg', 'LS GCV',  'Hybrid wgcv', 'GCV stop', 'Dual MIHS (svd)', 'Dual MIHS (gkl)')


