% Solution: reshape for transpose of what you need then transpose result!
clear all
dx = 2;
dy = 1;
dz = 3;
wc_arr = linspace(0.03, 1, 10);

%path = "runs/SaturatedVanDerPol/Supervised_noise/T_star/exp_10_0.1-2_ok3/zi_mesh_-11/";
path = "runs/Reversed_Duffing_Oscillator/Supervised_noise/T_star/exp_10_wc0.03-1_2/xzi_mesh/";
Darr = table2array(readtable(append(path, 'D_arr.csv')));
Darr = Darr(:, 2:end);

%%

% xbar = argsup(dT/dx (x_i))
% Tmax = dT/dx (xbar)
% zbar = argsup(dTstar/dz (z_i))
% Tstar_max = dTstar/dz (zbar)

% Criterion (analogy from transfer function of linear KKL):
% sup_{z, x} norm (  G(jw, z, x) =  (jwI - dTstar/dz(z) D dT/dx(x))^-1 dTstar/dz(z) F, infty)

figure()
hinf = zeros(length(wc_arr), 1);
Tmax_norm = zeros(length(wc_arr), 1);

for i = 1:length(wc_arr)
    wc = wc_arr(i);
    dTstardz = table2array(readtable(append(path, 'dTstar_dz_wc', sprintf('%0.2g', wc), '.csv')));
    dTstardz = dTstardz(:, 2:end);
    dTstardz = reshape(dTstardz, [length(dTstardz), dz, dx]);
    dTstardz = permute(dTstardz, [1, 3, 2]);
    dTdx = table2array(readtable(append(path, 'dTdx_wc', sprintf('%0.2g', wc), '.csv')));
    dTdx = dTdx(:, 2:end);
    dTdx = reshape(dTdx, [length(dTdx), dz, dx]);
    dTdx = permute(dTdx, [1, 3, 2]);
    D = reshape(Darr(i, :), [dz, dz]).'
    F = ones(length(D), dy); 
    % TODO
    hinf_z = zeros(length(dTdz), 1);
    for j = 1:length(dTdz)
        sys = ss(D, F, squeeze(dTdz(j, :, :)) * eye(length(D)), zeros(dx, dy));
        ninf = norm(sys, inf);
        hinf_z(j) = ninf;
    end
    [argvalue, argmax] = max(hinf_z);
    hinf(i) = argvalue;
end

N = 5e5 / length(wc_arr);
crit1 = hinf .* Tmax_norm;
figure()
plot(wc_arr, hinf)
hold on
plot(wc_arr, Tmax_norm)
hold on
plot(wc_arr, crit1)
legend('hinf','Tmax norm', 'crit1')

figure()
plot(wc_arr, crit1)
legend('crit1')

csvwrite(append(path, 'crit1.csv'), [wc_arr', Tmax_norm, hinf, crit1])