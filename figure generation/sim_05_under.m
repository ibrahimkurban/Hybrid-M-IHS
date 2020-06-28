%% OVER-DETERMIEND EXAMPLE 2 seismic

close all
clear all
clc

%% PARAMETERS
n               = 1500;
d               = 40000;
setup.N_sample  = 20;
setup.level     = repelem([0.003, 0.006 0.01 0.04 0.08 0.1 0.12 0.15], setup.N_sample);  
setup.N_noise   = numel(setup.level);

%problem parameters
setup.profile = 10;
setup.kappa   = 8;
setup.max     = 100;
setup.matrix_dev = 6;
setup.matrix_cor = 0.9;

%% GENERATE DATA
[A, b, setup.x0, x1, setup.dev0, U, setup.sigA, V] = generate_data(n,d,...
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
[setup.xxOR, parOR, setup.errOR, setup.err_rid] = oracle_methods(U, setup.sigA, V,b,setup.x0,x1);
err_x           = @(xx,i)(setup.errOR{1}(xx,i));
setup.k0        = parOR(1,:)';
setup.par0      = parOR(2,:)';
setup.Vx        = V'*setup.x0;



%% SOLVER parameters
setup.hy_iter   = 300;
setup.offset    = min(300, min(min(n,d)-setup.k0));
setup.gkl_size  = setup.offset+setup.k0;
setup.m         = 2*n;
setup.maxit     = 15;
setup.ptol      = 0;
setup.xtol1     = repelem([0.01, 0.01, 0.01, 0.01 0.01 0.01 0.01 0.01], setup.N_sample);

%% PREALLOCATION for MC simulation
N_met           = 6; %num of methods in the for bloop below
iter_noise      = zeros(N_met, setup.N_noise);
err_noise       = zeros(N_met, setup.N_noise);
par_noise       = zeros(N_met, setup.N_noise);
dev_noise       = zeros(N_met, setup.N_noise);
err_iter_noise  = cell(setup.N_noise,1);
par_iter_noise  = cell(setup.N_noise,1);
par2_iter_noise = cell(setup.N_noise,1);
gkl             = zeros(setup.N_noise,1);
ttt             = clock;
% MAIN LOOP
for i =1:setup.N_noise
    fprintf('Noise mc: %d/%d ...', i, setup.N_noise);
    tt      = clock;
    XTOL    = setup.xtol1(i);
    m       = setup.m;
    err     = zeros(N_met,1);
    pare    = zeros(N_met,1);
    deve    = zeros(N_met,1);
    iter    = ones(N_met,1);
    err_iter= cell(N_met,1);
    par_iter= cell(N_met,1);
    k       = 1;
    
    %% 1 ORACLE LS
    err(k)           = err_x(setup.xxOR{2}(:,i),i);
    pare(k)          = setup.par0(i);
    deve(k)          = setup.dev0(i);
    err_iter{k}      = err(k)*ones(setup.hy_iter,1);
    par_iter{k}      = pare(k)*ones(setup.hy_iter,1);
    k                = k+1;
    
    %% 2 LS GCV
    [x, part, devt]  = LS_gcv_iko(U, setup.sigA,V,b(:,i), x1);
    err(k)           = err_x(x,i);
    pare(k)          = part;
    deve(k)          = devt;
    err_iter{k}      = err(k)*ones(setup.hy_iter,1);
    par_iter{k}      = pare(k)*ones(setup.hy_iter,1);
    k                = k+1;
           
     
    %% 3-4 HYBRID
    options          = HyBRset_iko('RegPar', 'WGCV', 'Iter', setup.hy_iter, 'Omega', 'adapt', 'Reorth', 'on', 'x_true', setup.xxOR{1}(:,i));
    [~, out]         = HyBR_iko(A, b(:,i), [], options);
    
    %STOP
    iter(k)          = out.iterations;
    err(k)           = out.Enrm(out.iterations);
    pare(k)          = out.alpha.^2;
    err_iter{k}      = out.Enrm(1:out.iterations);
    par_iter{k}      = out.Alpha(1:out.iterations).^2;
    k                = k+1;
    
    %NON-STOP
    iter(k)          = setup.hy_iter;
    err(k)           = out.Enrm(end);
    pare(k)          = out.Alpha(end).^2;
    err_iter{k}      = out.Enrm;
    par_iter{k}      = out.Alpha.^2;
    k                = k+1;
    
    %% 5 HYBRID corrected ONE-SHOT
    [x, part, devt] =  hybrid_corrected_one_shot_iko(A,b(:,i),setup.gkl_size(i));
    err(k)           = err_x(x,i);
    pare(k)          = part;
    deve(k)          = devt;
    iter(k)          = setup.gkl_size(i);
    err_iter{k}      = err(k)*ones(setup.hy_iter,1);
    par_iter{k}      = pare(k)*ones(setup.hy_iter,1);
    k                = k+1;
    
    %% 6 PD M-IHS GKL SRHT
    [~, xx, part, out]  = reg_dual_mihs_gkl_iko(A,b(:,i),m,x1,XTOL,setup.ptol,setup.maxit);
    errs             = err_x(xx,i);
    err(k)           = errs(end);
    pare(k)          = part(end);
    iter(k)          = length(errs);
    err_iter{k}      = errs;
    par_iter{k}      = part;
    gkl(i,1)         = out.L;
    k                = k+1;
    
    
    %% save data
    iter_noise(:,i)  = iter;
    err_noise(:,i)   = err;
    par_noise(:,i)   = pare;
    dev_noise(:,i)   = deve;
    err_iter_noise{i}= err_iter;
    par_iter_noise{i}= par_iter;
    
    %% time
    tt2 = clock;
    fprintf(' %3.0f mins elapsed\n', (tt2(4) - tt(4))*60 + tt2(5) - tt(5))
end

clear A b U V out part devt part2 x iter err pare deve err_iter par_iter XTOL d errs i k m options xx
save(sprintf('sim_05_under_d%d%d_t%d%d', ttt(3), ttt(2), ttt(4), ttt(5)))



