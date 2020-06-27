function [x_hyc, par_hyc, dev_est_hyc] =  hybrid_corrected_one_shot_iko(A,b,k)
%%HYBRID_CORRECTED_ONE_SHOT implementation of the corrected hybrid method
%%that we mentioend in the article. Compute k dimensional LOWER bidiagonal
%%matrix and apply GCV on it by taking SVD. Also while finding the minimum
%%of the GCV functional, it uses findpeaks at selects last peak
%
% [x_hyc, par_hyc, dev_est_hyc] =  hybrid_corrected_one_shot_iko(A,b,k)
%
%
% ÝbrahimKurban Özaslan
% Bilkent EEE
% November 20019
%

%% GKL
[V,B] =gkl_hybrid_iko(A, b, k);

%% SVD
[Us,sigs,Vs] = dsvd(B);
e1 = zeros(k+1,1);
e1(1) = norm(b);

%% GCV
[y_hyc, par_hyc, dev_est_hyc]     = LS_gcv_iko_corrected(Us, sigs,Vs,e1, length(b));

%% Solution
x_hyc = V*y_hyc;

end



function [x_ls_gcv, par, dev_est,time, obj, lambdas] = LS_gcv_iko_corrected(U, sig,V,b, nn)
%%LS_GCV calcualtes RegLS solutionwith paramter found via GCV of G.Golub.
%
%   Author  : Ibrahim Kurban Ozaslan
%   Date    : 14.07.2018
%   v2      : finds lambda by fminbnd
%
%   [x_ls_gcv, par_ls_gcv, obj_ls_gcv_par, lambdas_ls_gcv] = LS_gcv(A,b, x1)
%
%   x1 : prior
%

tic;
%% take SVD of matrix and variables
sig2    = sig.^2;

Ub      = U'*b;
g       = Ub;
g2      = g.^2;
delta2  = norm(b)^2 - norm(Ub)^2;
if((delta2) < 1e-14)
    delta2 = 0;
end
%% function to minimize
d       = length(sig);

MN      = (nn-d).*(nn>d);
beta    = @(lambda)( lambda./(sig2 + lambda));
obj     = @(lambda)((sum((beta(lambda).^2).*g2) + delta2)...
    /(MN + sum(beta(lambda)))^2);

% %% Parameter Grid iteration
% if(nargout <= 3)
%     %minimization
%     options             = optimset('TolX', 1e-4, 'Display','off');
%     [par_log,gcv_val]          = fminbnd(@(lam)obj(10^lam), -14, log10(sig(1)), options);
%     par = 10^par_log;
% else
%% Search Grid num of points at 1. and 2. iterate
grid1       = 200;
grid2       = 100;
lambdas1    = logspace(-16, log10(max(sig)), grid1);
[~, ~, obj, lambdas] = grid_search(obj, lambdas1, grid2);
[~, indi]   = findpeaks(-log10(obj));
ind         = max(indi);
par         = lambdas(ind);
gcv_val     = obj(ind);
% end
dev_est = sqrt(gcv_val)*sqrt(MN + sum(beta(par)));
%% solution
x_ls_gcv     = V*( (sig./(sig2 + par)).*g );
time = toc;
end






