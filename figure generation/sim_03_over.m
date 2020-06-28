%% OVER-DETERMIEND EXAMPLE 1 xray tomo

close all
clear all
clc

%% PARAMETERS
setup.n         = 50;
setup.N_sample  = 20;
setup.level     = repelem([0.003, 0.006 0.01 0.04 0.08 0.1 0.12 0.15], setup.N_sample);  
setup.N_noise   = numel(setup.level);

%problem parameters
setup.problem   = 3;
setup.image     = 1;
setup.type      = 1;
setup.par1      = round(sqrt(2)*setup.n);
setup.par_name  = 'p';

%% GENERATE DATA
[A, b, x1, setup.x0, err_x0, setup.dev0, setup.prob_info, err_x, setup.k0, setup.par0, setup.xxOR, U, setup.sigA, V] ...
                = generate_data_IRtool_iko(setup.n, setup.problem, setup.image, setup.level, 1, 1, 'type', setup.type, 'p', setup.par1);
setup.size      = size(A);
[n,d]           = size(A);

%% SOLVER parameters
setup.hy_iter   = 300;
setup.offset    = min(500, min(min(n,d)-setup.k0));
setup.gkl_size  = setup.offset+setup.k0;
setup.m         = 2*d;
setup.maxit     = 30;
setup.ptol      = 0;
setup.xtol1     = repelem([0.001, 0.001, 0.001, 0.001 0.001 0.001 0.001 0.001], setup.N_sample);
setup.Vx        = V'*setup.x0;

%% PREALLOCATION for MC simulation
N_met           = 8; %num of methods in the for bloop below
iter_noise      = zeros(N_met, setup.N_noise);
err_noise       = zeros(N_met, setup.N_noise);
par_noise       = zeros(N_met, setup.N_noise);
dev_noise       = zeros(N_met, setup.N_noise);
err_iter_noise  = cell(setup.N_noise,1);
par_iter_noise  = cell(setup.N_noise,1);
par2_iter_noise = cell(setup.N_noise,1);
gkl             = zeros(2,setup.N_noise);
Utb             = U'*b;
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
    gkll    = zeros(2,1);
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
        
    %% 3 LS GCV eff
    [x, part, devt]  = LS_gcv_iko(speye(d), setup.sigA,V, Utb(:,i), x1);
    err(k)           = err_x(x,i);
    pare(k)          = part;
    deve(k)          = devt;
    iter(k)          = d;
    err_iter{k}      = err(k)*ones(setup.hy_iter,1);
    par_iter{k}      = pare(k)*ones(setup.hy_iter,1);
    k                = k+1;    
      
    
    %% 4-5 HYBRID
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
    
    %% 6 HYBRID corrected ONE-SHOT
    [x, part, devt] =  hybrid_corrected_one_shot_iko(A,b(:,i),setup.gkl_size(i));
    err(k)           = err_x(x,i);
    pare(k)          = part;
    deve(k)          = devt;
    iter(k)          = setup.gkl_size(i);
    err_iter{k}      = err(k)*ones(setup.hy_iter,1);
    par_iter{k}      = pare(k)*ones(setup.hy_iter,1);
    k                = k+1;
    
    %% 7 PD M-IHS GKL SRHT
    [~, xx, part, out]  = reg_mihs_gkl_lower_iko(A,b(:,i),m,x1,XTOL,setup.ptol,setup.maxit);
    errs             = err_x(xx,i);
    err(k)           = errs(end);
    pare(k)          = part(end);
    iter(k)          = length(errs);
    err_iter{k}      = errs;
    par_iter{k}      = part;
    gkll(1)          = out.L;
    k                = k+1;
    
    %% 8 PD M-IHS GKL COUNTSKETCH
    %Count sketch generation
    params              = struct();
    params.SA           = generate_SA_count([A, b(:,i)], m);
    params.Sb           = params.SA(:,end);
    params.SA(:,end)    = [];
    %sovler
    [~, xx, part, out]  = reg_mihs_gkl_lower_iko(A,b(:,i),m,x1,XTOL,setup.ptol,setup.maxit,params);
    errs             = err_x(xx,i);
    err(k)           = errs(end);
    pare(k)          = part(end);
    iter(k)          = length(errs);
    err_iter{k}      = errs;
    par_iter{k}      = part;
    gkll(2)          = out.L;
    k                = k+1;
    
    %% save data
    iter_noise(:,i)  = iter;
    err_noise(:,i)   = err;
    par_noise(:,i)   = pare;
    dev_noise(:,i)   = deve;
    err_iter_noise{i}= err_iter;
    par_iter_noise{i}= par_iter;
    gkl(:,i)         = gkll;
    
    %% time
    tt2 = clock;
    fprintf(' %3.0f mins elapsed\n', (tt2(4) - tt(4))*60 + tt2(5) - tt(5))
end

clear A b U V out part devt part2 x iter err pare deve err_iter par_iter XTOL d errs i k m options xx Utb SA Sb Us sigs Vs
save(sprintf('sim_03_over_d%d%d_t%d%d', ttt(3), ttt(2), ttt(4), ttt(5)))



