% Solution: reshape for transpose of what you need then transpose result!
clear all
dx = 2;
dy = 1;
dz = 3;
wc_arr = linspace(0.03, 1, 100);

%path = "runs/VanDerPol/Supervised_noise/T_star/exp_100_wc0.03-1_-11+1cycle_rk41e-2/xzi_mesh/";
path = "runs/Reversed_Duffing_Oscillator/Supervised_noise/T_star/exp_100_wc0.03-1_rk41e-3_k10/xzi_mesh/";
Darr = table2array(readtable(append(path, 'D_arr.csv')));
Darr = Darr(:, 2:end);

%%

% xbar = argsup(dT/dx (x_i))
% Tmax = dT/dx (xbar)
% zbar = argsup(dTstar/dz (z_i))
% Tstar_max = dTstar/dz (zbar)

% Criterion 1 (analogy from transfer function of linear KKL):
% norm ( G(jw, z, x) =  (jwI - Tstar_max D Tmax)^-1 Tstar_max F, infty)

figure()
hinf = zeros(length(wc_arr), 1);
Tmax_norm = zeros(length(wc_arr), 1);
Tstar_max_norm = zeros(length(wc_arr), 1);

for i = 1:length(wc_arr)
    wc = wc_arr(i);
    Tstar_max = table2array(readtable(append(path, 'Tstar_max_wc', sprintf('%0.2g', wc), '.csv')));
    Tstar_max = Tstar_max(:, 2:end)
    Tstar_max_norm(i) = norm(Tstar_max, 2);
    Tmax = table2array(readtable(append(path, 'Tmax_wc', sprintf('%0.2g', wc), '.csv')));
    Tmax = Tmax(:, 2:end)
    Tmax_norm(i) = norm(Tmax, 2);
    D = reshape(Darr(i, :), [dz, dz]).'
    F = ones(dz, dy);
    eig(Tstar_max * D * Tmax)  % Not stable?
    sys = ss(Tstar_max * D * Tmax, Tstar_max * F, eye(dx), zeros(dx, dy));
    bode(sys(1, 1))
    hold on
    ninf = norm(sys, inf);
    hinf(i) = ninf;
end

N = 5e5 / length(wc_arr);

crit1 = hinf;
h = figure();
plot(wc_arr, Tmax_norm)
hold on
plot(wc_arr, Tstar_max_norm)
hold on
plot(wc_arr, crit1)
legend('Tmax norm', 'Tstar max norm', 'crit1')
savefig(h, append(path, 'crit1.fig'))

figure()
plot(wc_arr, crit1)
legend('crit1')

csvwrite(append(path, 'crit1.csv'), [wc_arr', Tmax_norm, Tstar_max_norm, crit1])

%%

% Criterion 2 (analogy from transfer function of linear KKL):
% sup_{z, x} norm (G(jw, z, x) =  (jwI - dTstar/dz(z) D dT/dx(x))^-1 dTstar/dz(z) F, infty)

hinf = zeros(length(wc_arr), 1);
Tmax_norm = zeros(length(wc_arr), 1);
Tstar_max_norm = zeros(length(wc_arr), 1);

for i = 1:length(wc_arr)
    wc = wc_arr(i);
    dTstardz = table2array(readtable(append(path, 'dTstar_dz_wc', sprintf('%0.2g', wc), '.csv')));
    dTstardz = dTstardz(:, 2:end);
    dTstardz = reshape(dTstardz, [length(dTstardz), dz, dx]);
    dTstardz = permute(dTstardz, [1, 3, 2]);
    Tmax_norm(i) = norm(dTstardz(:), 2);
    dTdx = table2array(readtable(append(path, 'dTdx_wc', sprintf('%0.2g', wc), '.csv')));
    dTdx = dTdx(:, 2:end);
    dTdx = reshape(dTdx, [length(dTdx), dx, dz]);
    dTdx = permute(dTdx, [1, 3, 2]);
    Tmax_norm(i) = norm(dTdx(:), 2);
    D = reshape(Darr(i, :), [dz, dz]).'
    F = ones(dz, dy);
    hinf_z = zeros(length(dTdx), 1);
    for j = 1:length(dTdx)
        current_dTstardz = squeeze(dTstardz(j, :, :));
        current_dTdx = squeeze(dTdx(j, :, :));
        eigvals = eig(current_dTstardz * D * current_dTdx);  % Not stable?
        sys = ss(current_dTstardz * D * current_dTdx, current_dTstardz * F, eye(dx), zeros(dx, dy));
        ninf = norm(sys, inf);
        hinf_z(j) = ninf;
    end
    eigvals
    [argvalue, argmax] = max(hinf_z);
    hinf(i) = argvalue;
end

N = 5e5 / length(wc_arr);
crit2 = hinf;
h = figure();
plot(wc_arr, Tmax_norm)
hold 
plot(wc_arr, Tstar_max_norm)
hold on
plot(wc_arr, crit2)
legend('T norm', 'Tstar norm', 'crit2')
savefig(h, append(path, 'crit2.fig'))

figure()
plot(wc_arr, crit2)
legend('crit2')

csvwrite(append(path, 'crit2.csv'), [wc_arr', Tmax_norm, Tstar_max_norm, crit2])

%%

% Criterion 3 (analogy from transfer function of linear KKL):
% norm (G(jw, z, x) =  (jwI - mean(dTstar/dz(z)) D mean(dT/dx(x)))^-1 mean(dTstar/dz(z)) F, infty)

hinf = zeros(length(wc_arr), 1);
Tmax_norm = zeros(length(wc_arr), 1);
Tstar_max_norm = zeros(length(wc_arr), 1);

for i = 1:length(wc_arr)
    wc = wc_arr(i);
    dTstardz = table2array(readtable(append(path, 'dTstar_dz_wc', sprintf('%0.2g', wc), '.csv')));
    dTstardz = dTstardz(:, 2:end);
    dTstardz = reshape(dTstardz, [length(dTstardz), dz, dx]);
    dTstardz = permute(dTstardz, [1, 3, 2]);
    Tstar_mean = squeeze(mean(dTstardz(:, :, :)));
    Tstar_max_norm(i) = norm(dTstardz(:), 2);
    dTdx = table2array(readtable(append(path, 'dTdx_wc', sprintf('%0.2g', wc), '.csv')));
    dTdx = dTdx(:, 2:end);
    dTdx = reshape(dTdx, [length(dTdx), dx, dz]);
    dTdx = permute(dTdx, [1, 3, 2]);
    Tmean = squeeze(mean(dTdx(:, :, :)));
    Tmax_norm(i) = norm(dTdx(:), 2);
    D = reshape(Darr(i, :), [dz, dz]).'
    F = ones(dz, dy);
    eig(Tstar_mean * D * Tmean)  % Not stable?
    sys = ss(Tstar_mean * D * Tmean, Tstar_mean * F, eye(dx), zeros(dx, dy));
    bode(sys(1, 1))
    hold on
    ninf = norm(sys, inf);
    hinf(i) = ninf;
end

N = 5e5 / length(wc_arr);
crit3 = hinf;
h = figure();
plot(wc_arr, Tmax_norm)
hold 
plot(wc_arr, Tstar_max_norm)
hold on
plot(wc_arr, crit3)
legend('T norm', 'Tstar norm', 'crit3')
savefig(h, append(path, 'crit3.fig'))

figure()
plot(wc_arr, crit3)
legend('crit3')

csvwrite(append(path, 'crit3.csv'), [wc_arr', Tmax_norm, Tstar_max_norm, crit3])