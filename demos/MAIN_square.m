clear all
close all
clc

%% some parameters
n           = 30;%100
noise_level = 0.01;

%% GENERATE DATA 1
[A, b, x1, x0, err_x0, dev, prob_info, err_x, k0, par0, xxOR, U, sigA, V] ...
            = generate_data_IRtool_iko(n, 1, 9, noise_level, 1, 1, 'type', 1, 'level',2);

%% GENERATE DATA 2
% [A, b, x1, x0, err_x0, dev, prob_info, err_x, k0, par0, xxOR, U, sigA, V] ...
%             = generate_data_IRtool_iko(n, 2, 8, noise_level, 1, 1, 'type', 2, 'level', 1, 'p', 2*n);
%% error metric etc.
[n,d] = size(A);
% xreg        = xxOR{2};
% xtrun       = V(:,1:k0)*((sigA(1:k0).^-1).*(U(:,1:k0)'*b));
% err_xtrun   = @(xx)(sqrt(sum((xtrun - xx).^2, 1))/norm(xtrun));
% err_xreg    = @(xx)(sqrt(sum((xreg - xx).^2, 1))/norm(xreg));

%% Full LS Soution
L                   = min([k0+500, n,d]);
[x_gcv, par_gcv]    = LS_gcv_iko(U, sigA,V,b, x1);
[x_gcv2, par_gcv2]  = LS_gcv_iko(eye(L), sigA(1:L),V(:, 1:L), U(:,1:L)'*b, x1);
fprintf('Reg LS GCV-full reg. par       : %1.2e\n', par_gcv);
fprintf('Reg LS GCV-partial reg. par    : %1.2e\n', par_gcv2);
%% Hybrid
HYBRID_iter      = 300;
options          = HyBRset_iko('RegPar', 'WGCV', 'Iter', HYBRID_iter, 'Omega', 'adapt', 'Reorth', 'on', 'x_true', xxOR{1});
fprintf('\nHyBR runs...'); tic;
[x_hy, output] = HyBR_iko(A, b, [], options);
fprintf('%2.1e sec elapsed\n\n', toc);

%% METHODS
XTOL    = [1e-3 0];
MAXIT   = [15 25];
PTOL    = [0 0];
m1      = 2*k0;
m2      = 5*k0;
L       = min(k0+500, m1);

[~, xx1, pari1, pars1]      = reg_pd_mihs_svd_lower_iko(  A,b,[m1, m2],x1,XTOL,PTOL, MAXIT);
[~, xx2, pari2, pars2]      = reg_pd_mihs_gkl_lower_iko(  A,b,[m1, m2],x1,XTOL,PTOL, MAXIT, struct('L', L));

%% plot
figure;
hold on; grid on;
xlabel('iterate, k'); ylabel('oracle error');
plot(log10(err_x(xxOR{2}))*ones(15,1), 'k:', 'linewidth', 2);
plot(log10(err_x(x_gcv))*ones(15,1), 'r:', 'linewidth', 2);
plot(log10(output.Enrm(1:end)), '-.', 'linewidth', 2);
plot(output.iterations, log10(output.Enrm(output.iterations)), 'o', 'markersize', 10)
plot(log10(err_x(xx1)), 'linewidth', 2);
plot(log10(err_x(xx2)), 'linewidth', 2);
legend('Oracle Reg', 'LS GCV', 'HYBRID', 'HYBRID stop by GCV', 'M-IHS svd', 'M-IHS gkl')

figure;
hold on; grid on;
xlabel('iterate, k'); ylabel('\lambda');
plot(log10(par0)*ones(15,1), 'k:', 'linewidth', 2);
plot(log10(par_gcv)*ones(15,1), 'r:', 'linewidth', 2);
plot(log10(output.Alpha.^2), '-.', 'linewidth', 2)
plot(log10(pari1(:, end)), 'linewidth', 2);
plot(log10(pari2(:,end)), 'linewidth', 2);
legend('Oracle Reg', 'LS GCV', 'HYBRID','PD-gkl', 'PD svd')

