% Matlab reshapes opposite of python, by columns and not by rows!! 
% Solution: reshape for transpose of what you need then transpose result!
clear all
dx = 2;
dy = 1;
dz = 3;
wc_arr = linspace(0.1, 2., 200);
%dx = 4;
%dy = 1;
%dz = 5;
%wc_arr = linspace(1, 5., 50);

%path = "runs/VanDerPol/Supervised_noise/T_star/exp_100_wc0.03-1_-11+1cycle_rk41e-2/xzi_mesh/";
path = "runs/Reversed_Duffing_Oscillator/Supervised_noise/T_star/exp200_DoptimAE05_wc01-2_rk41e-3_k10/xzi_mesh/";
%path = "runs/SaturatedVanDerPol/Supervised_noise/T_star/exp_100_wc0.03-1_-2727_rk41e-3_2/xzi_mesh/";
%path = "runs/QuanserQubeServo2_meas1/Supervised_noise/T_star/Ntraj5000_wc1550/xzi_mesh/";
Darr = table2array(readtable(append(path, 'D_arr.csv')));
Darr = Darr(:, 2:end);

%%

% Criterion: norm(vector(norm(dTstar/dz(z_i), 2)_i, 2) * (sup(G_epsilon(jw)) + norm(G_z, 2))

figure()
hinf = zeros(length(wc_arr), 1);
hinf_z = zeros(length(wc_arr), 1);
dTdz_norm = zeros(length(wc_arr), 1);
N = 10000;

for i = 1:length(wc_arr)
    wc = wc_arr(i);
    dTdz = table2array(readtable(append(path, 'dTstar_dz_wc', sprintf('%0.2g', wc), '.csv')));
    dTdz = dTdz(:, 2:end);
    dTdz = reshape(dTdz, [length(dTdz), dz, dx]);
    dTdz = permute(dTdz, [1, 3, 2]);
    vector_dTdz_norm = zeros(length(dTdz), 1);
    for j = 1:length(dTdz)
        vector_dTdz_norm(j) = norm(squeeze(dTdz(j, :, :)), 2);
    end
    dTdz_norm(i) = norm(vector_dTdz_norm, 2);
    D = reshape(Darr(i, :), [dz, dz]).'
    F = ones(dz, dy);

    sys = ss(D, F, eye(length(D)), zeros(dz, dy));
    bode(sys)
    hold on
%     [ninf,fpeak] = hinfnorm(sys);
    ninf = norm(sys, inf);
    hinf(i) = ninf;

    sys_z = ss(D, eye(dz), eye(dz), zeros(dz, dz));
    % bode(sys_z)
    % hold on
    n2_z = norm(sys_z, 2);
    hinf_z(i) = n2_z;
end

crit = (hinf + hinf_z) .* dTdz_norm;
h = figure
plot(wc_arr, hinf)
hold on
plot(wc_arr, hinf_z)
hold on
plot(wc_arr, dTdz_norm)

hold on
plot(wc_arr, crit)
legend('hinf', 'hinf z', 'dTdz norm', 'crit')
savefig(h, append(path, 'crit.fig'))

figure()
plot(wc_arr, crit)
legend('crit')

csvwrite(append(path, 'crit.csv'), [wc_arr', dTdz_norm, hinf, hinf_z, crit])