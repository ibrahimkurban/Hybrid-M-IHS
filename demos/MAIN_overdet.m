clear all
close all
clc
%% some parameters
n           = 30;
noise_level = 0.01;

%% GENERATE DATA 1
% [A, b, x1, x0, err_x0, dev, prob_info, err_x, k0, par0, xxOR, U, sigA, V] ...
%             = generate_data_IRtool_iko(n, 3, 1, noise_level, 1, 1, 'type', 1, 'p',round(sqrt(2)*n));

%% GENERATE DATA 2
[A, b, x1, x0, err_x0, dev, prob_info, err_x, k0, par0, xxOR, U, sigA, V] ...
            = generate_data_IRtool_iko(n, 2, 8, noise_level, 1, 1, 'type', 1, 'level', 4, 'p', 10*n);

%% error metric etc.
[n,d] = size(A);
% xreg        = xxOR{2};
% xtrun       = V(:,1:k0)*((sigA(1:k0).^-1).*(U(:,1:k0)'*b));
% err_xtrun   = @(xx)(sqrt(sum((xtrun - xx).^2, 1))/norm(xtrun));
% err_xreg    = @(xx)(sqrt(sum((xreg - xx).^2, 1))/norm(xreg));

%% Full LS Soution
[x_gcv, par_gcv, dev_est]     = LS_gcv_iko(U, sigA,V,b, x1);
[x_gcv2, par_gcv2, dev_est2]  = LS_gcv_iko(speye(d), sigA,V,U'*b, x1);

fprintf('Reg LS GCV-full reg. par            : %1.2e\n', par_gcv);
fprintf('Reg LS GCV-partial reg. par         : %1.2e\n', par_gcv2);


%% Hybrid -LSQR
HYBRID_iter     = 300;
options         = HyBRset_iko('RegPar', 'WGCV', 'Iter', HYBRID_iter, 'Omega', 'adapt','Reorth', 'on', 'x_true', xxOR{1});
fprintf('\nHyBR runs...'); tic;
[x_hy, output] = HyBR_iko(A, b, [], options);
fprintf('%2.1e sec elapsed\n\n', toc);


%% RP
fprintf('RP runs...\n')
m           = 2*d;
% [SA, rpt1]  = generate_SA_count([A, b,],m);
[SA, rpt1]  = generate_SA_iko([A, b],m);
SA          = full(SA);
Sb          = SA(:,end);
SA          = SA(:,1:end-1);
fprintf('finished\n')

%% METHODS
XTOL    = 0;
MAXIT   = 30+ ceil(log(max(size(A))));
PTOL    = 0;
L       = min([k0+500, d, n, m]);
% L=d;
params.SA = SA;
params.Sb = Sb;
[~, xx1, pari1]  = reg_mihs_svd_lower_iko(A,b,m,x1,XTOL,PTOL,MAXIT,params);
[~, xx2, pari2]  = reg_mihs_gkl_lower_iko(A,b,m,x1,XTOL,PTOL,MAXIT,struct('L', L));

%% plot
figure;
hold on; grid on;
xlabel('iterate, k'); ylabel('oracle error');
plot(log10(err_x(xxOR{2}))*ones(HYBRID_iter,1), 'k:', 'linewidth', 2);
plot(log10(err_x(x_gcv))*ones(HYBRID_iter,1), 'r:', 'linewidth', 2);
plot(log10(err_x(x_gcv2))*ones(HYBRID_iter,1), 'g:', 'linewidth', 3);
plot(log10(output.Enrm(1:end)), '-.', 'linewidth', 2);
plot(output.iterations, log10(output.Enrm(output.iterations)), 'o', 'markersize', 10)
plot(log10(err_x(xx1)), 'linewidth', 2);
plot(log10(err_x(xx2)), 'linewidth', 2);
legend('Oracle Reg', 'LS GCV', 'LS GCV(eff)', 'HYBRID', 'HYBRID stop by GCV', 'M-IHS (svd)', 'M-IHS (GKL)')

figure;
hold on; grid on;
xlabel('iterate, k'); ylabel('\lambda');
plot(log10(par0)*ones(HYBRID_iter,1), 'k:', 'linewidth', 2);
plot(log10(par_gcv)*ones(HYBRID_iter,1), 'r:', 'linewidth', 2);
plot(log10(par_gcv2)*ones(HYBRID_iter,1), 'g:', 'linewidth', 3);
plot(log10(output.Alpha)*2, '-.', 'linewidth', 2)
plot(output.iterations, 2*log10(output.Alpha(output.iterations)), 'o', 'markersize', 10)
plot(log10(pari1(:, end)), 'linewidth', 2);
plot(log10(pari2(:,end)), 'linewidth', 2);
legend('Oracle Reg', 'LS GCV', 'LS GCV(eff)', 'HYBRID', 'HYBRID stop by GCV', 'M-IHS (svd)', 'M-IHS (GKL)')

