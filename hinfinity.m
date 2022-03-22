
% Matlab reshapes opposite of python, by columns and not by rows!! 
% Solution: reshape for transpose of what you need then transpose result!
clear all
dx = 2;
dy = 1;
dz = 3;
wc_arr = linspace(0.03, 1, 100);

%path = "runs/VanDerPol/Supervised_noise/T_star/exp_100_wc0.03-1_-11+1cycle_rk41e-2/xzi_mesh/";
path = "runs/Reversed_Duffing_Oscillator/Supervised_noise/T_star/exp_100_wc0.03-1_rk41e-3_k10_2/xzi_mesh/";
Darr = table2array(readtable(append(path, 'D_arr.csv')));
Darr = Darr(:, 2:end);

%%

% zbar = argsup(dTstar/dz (z_i))
% Tmax = dTstar/dz (zbar)

% Criterion 1: norm(Tmax) * sup(G(jw))

figure()
hinf = zeros(length(wc_arr), 1);
Tmax_norm = zeros(length(wc_arr), 1);

for i = 1:length(wc_arr)
    wc = wc_arr(i);
    Tmax = table2array(readtable(append(path, 'Tstar_max_wc', sprintf('%0.2g', wc), '.csv')));
    %Tmax = table2array(readtable(append(path, 'Tmax_wc', sprintf('%0.2g', wc), '.csv')));
    %Tmax = table2array(readtable(append(path, 'Tmax_wc', sprintf('%0.2g', round(wc, 2)), '.csv')));
    %Tmax = table2array(readtable(append(path, 'Tmax_wc', sprintf('%0.3g', round(wc, 2)), '.csv')));
    %Tmax = table2array(readtable(append(path, 'Tmax_wc', sprintf('%0.3g', wc), '.csv')));
    Tmax = Tmax(:, 2:end)
    Tmax_norm(i) = norm(Tmax, 2);
    D = reshape(Darr(i, :), [dz, dz]).'
    F = ones(length(D), dy); 
    sys = ss(D, F, eye(length(D)), zeros(length(D), dy));
    bode(sys(1, 1))
    hold on
%     [ninf,fpeak] = hinfnorm(sys);
    ninf = norm(sys, inf);
    hinf(i) = ninf;
end

N = 5e5 / length(wc_arr);
crit1 = hinf .* Tmax_norm;
h = figure()
plot(wc_arr, hinf)
hold on
plot(wc_arr, Tmax_norm)
hold on
plot(wc_arr, crit1)
legend('hinf','Tmax norm', 'crit1')
savefig(h, append(path, 'crit1_old.fig'))


figure()
plot(wc_arr, crit1)
legend('crit1')

csvwrite(append(path, 'crit1_old.csv'), [wc_arr', Tmax_norm, hinf, crit1])

%%

% Criterion 2: sup(Tmax * G(jw))

figure()
hinf = zeros(length(wc_arr), 1);
Tmax_norm = zeros(length(wc_arr), 1);

for i = 1:length(wc_arr)
    wc = wc_arr(i);
    Tmax = table2array(readtable(append(path, 'Tstar_max_wc', sprintf('%0.2g', wc), '.csv')));
    %Tmax = table2array(readtable(append(path, 'Tmax_wc', sprintf('%0.1f', wc), '.csv')));
    %Tmax = table2array(readtable(append(path, 'Tmax_wc', sprintf('%0.2g', wc), '.csv')));
    %Tmax = table2array(readtable(append(path, 'Tmax_wc', sprintf('%0.2g', round(wc, 2)), '.csv')));
    %Tmax = table2array(readtable(append(path, 'Tmax_wc', sprintf('%0.3g', round(wc, 2)), '.csv')));
    %Tmax = table2array(readtable(append(path, 'Tmax_wc', sprintf('%0.3g', wc), '.csv')));
    Tmax = Tmax(:, 2:end)
    Tmax_norm(i) = norm(Tmax, 2);
    D = reshape(Darr(i, :), [dz, dz]).'
    F = ones(length(D), dy); 
    sys = ss(D, F, Tmax * eye(length(D)), zeros(dx, dy));
    bode(sys(1, 1))
    hold on
    ninf = norm(sys, inf);
    hinf(i) = ninf;
end

N = 5e5 / length(wc_arr);
crit2 = hinf;
figure()
plot(wc_arr, hinf)
hold on
plot(wc_arr, Tmax_norm)
hold on
plot(wc_arr, crit2)
legend('hinf', 'Tmax_norm', 'crit2')

figure()
plot(wc_arr, crit2)
legend('crit2')

csvwrite(append(path, 'crit2.csv'), [wc_arr', Tmax_norm, hinf, crit2])


%%

% Criterion 3: sup_{z_i, w} (dTstar/dz (z_i) G(jw))

hinf = zeros(length(wc_arr), 1);

for i = 1:length(wc_arr)
    wc = wc_arr(i);
    %dTdz = table2array(readtable(append(path, 'dTdz_wc', sprintf('%0.1f', wc), '.csv')));
    dTdz = table2array(readtable(append(path, 'dTdz_wc', sprintf('%0.2g', wc), '.csv')));
    %dTdz = table2array(readtable(append(path, 'dTdz_wc', sprintf('%0.2g', round(wc, 2)), '.csv')));
    %dTdz = table2array(readtable(append(path, 'dTdz_wc', sprintf('%0.3g', round(wc, 2)), '.csv')));
    %dTdz = table2array(readtable(append(path, 'dTdz_wc', sprintf('%0.3g', wc), '.csv')));
    dTdz = dTdz(:, 2:end);
    dTdz = reshape(dTdz, [length(dTdz), dz, dx]);
    dTdz = permute(dTdz, [1, 3, 2]);
    %[argvalue, argmax] = max(vecnorm(dTdz(:, :), 2, 2))
    D = reshape(Darr(i, :), [dz, dz]).'
    F = ones(length(D), dy);
    % no other choice than for loop due to Matlab not handling
    % multidimensional arrays well, and arrays of systems?
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
crit3 = hinf;
figure()
plot(wc_arr, hinf)
hold on
plot(wc_arr, crit3)
legend('hinf', 'crit3')

figure()
plot(wc_arr, crit3)
legend('crit3')

csvwrite(append(path, 'crit3.csv'), [wc_arr', hinf, crit3])

%%

% Criterion 4: norm(dTstar/dz, 2) * sup(G(jw))
% Same as criterion 1 but norm l2 of whole dTdz over all z_i

figure()
hinf = zeros(length(wc_arr), 1);
Tmax_norm = zeros(length(wc_arr), 1);

for i = 1:length(wc_arr)
    wc = wc_arr(i);
    dTdz = table2array(readtable(append(path, 'dTstar_dz_wc', sprintf('%0.2g', wc), '.csv')));
    %dTdz = table2array(readtable(append(path, 'dTdz_wc', sprintf('%0.2g', wc), '.csv')));
    %dTdz = table2array(readtable(append(path, 'dTdz_wc', sprintf('%0.2g', round(wc, 2)), '.csv')));
    %dTdz = table2array(readtable(append(path, 'dTdz_wc', sprintf('%0.3g', round(wc, 2)), '.csv')));
    %dTdz = table2array(readtable(append(path, 'dTdz_wc', sprintf('%0.3g', wc), '.csv')));
    dTdz = dTdz(:, 2:end);
    dTdz = reshape(dTdz, [length(dTdz), dz, dx]);
    dTdz = permute(dTdz, [1, 3, 2]);
    Tmax_norm(i) = norm(dTdz(:), 2);
    D = reshape(Darr(i, :), [dz, dz]).'
    F = ones(length(D), dy); 
    sys = ss(D, F, eye(length(D)), zeros(length(D), dy));
    bode(sys)
    hold on
%     [ninf,fpeak] = hinfnorm(sys);
    ninf = norm(sys, inf);
    hinf(i) = ninf;
end

N = 5e5 / length(wc_arr);
crit4 = hinf .* Tmax_norm;
h = figure()
plot(wc_arr, hinf)
hold on
plot(wc_arr, Tmax_norm)
hold on
plot(wc_arr, crit4)
legend('hinf','Tmax norm', 'crit4')
savefig(h, append(path, 'crit4_old.fig'))

figure()
plot(wc_arr, crit4)
legend('crit4')

csvwrite(append(path, 'crit4_old.csv'), [wc_arr', Tmax_norm, hinf, crit4])

%%

% Criterion 5: norm(dTstar/dz, 2) * norm(G(jw), 2)
% Same as criterion 1 but norm l2 of whole dTdz over all z_i and norm 2 of
% linear system

figure()
hinf = zeros(length(wc_arr), 1);
Tmax_norm = zeros(length(wc_arr), 1);

for i = 1:length(wc_arr)
    wc = wc_arr(i);
    dTdz = table2array(readtable(append(path, 'dTdz_wc', sprintf('%0.2g', wc), '.csv')));
    %dTdz = table2array(readtable(append(path, 'dTdz_wc', sprintf('%0.2g', round(wc, 2)), '.csv')));
    %dTdz = table2array(readtable(append(path, 'dTdz_wc', sprintf('%0.3g', round(wc, 2)), '.csv')));
    %dTdz = table2array(readtable(append(path, 'dTdz_wc', sprintf('%0.3g', wc), '.csv')));
    dTdz = dTdz(:, 2:end);
    dTdz = reshape(dTdz, [length(dTdz), dz, dx]);
    dTdz = permute(dTdz, [1, 3, 2]);
    Tmax_norm(i) = norm(dTdz(:), 2);
    D = reshape(Darr(i, :), [dz, dz]).'
    F = ones(length(D), dy); 
    sys = ss(D, F, eye(length(D)), zeros(length(D), dy));
    bode(sys(1, 1))
    hold on
%     [ninf,fpeak] = hinfnorm(sys);
    ninf = norm(sys, 2);
    hinf(i) = ninf;
end

N = 5e5 / length(wc_arr);
crit5 = hinf .* Tmax_norm;
figure()
plot(wc_arr, hinf)
hold on
plot(wc_arr, Tmax_norm)
hold on
plot(wc_arr, crit5)
legend('hinf','Tmax norm', 'crit5')

figure()
plot(wc_arr, crit5)
legend('crit5')

csvwrite(append(path, 'crit5.csv'), [wc_arr', Tmax_norm, hinf, crit5])

%%

% Criterion 6: same as criterion 1 (norm(Tmax) * sup(G(jw))) but with normalized G (static gain 0)

figure()
hinf = zeros(length(wc_arr), 1);
Tmax_norm = zeros(length(wc_arr), 1);

for i = 1:length(wc_arr)
    wc = wc_arr(i);
    Tmax = table2array(readtable(append(path, 'Tmax_wc', sprintf('%0.2g', wc), '.csv')));
    %Tmax = table2array(readtable(append(path, 'Tmax_wc', sprintf('%0.2g', round(wc, 2)), '.csv')));
    %Tmax = table2array(readtable(append(path, 'Tmax_wc', sprintf('%0.3g', round(wc, 2)), '.csv')));
    %Tmax = table2array(readtable(append(path, 'Tmax_wc', sprintf('%0.3g', wc), '.csv')));
    Tmax = Tmax(:, 2:end)
    Tmax_norm(i) = norm(Tmax, 2);
    D = reshape(Darr(i, :), [dz, dz]).'
    F = ones(length(D), dy);
    G0 = -inv(D) * F ;
    sys = ss(D, F, eye(length(D)) ./ G0, zeros(length(D), dy));
    bode(sys(1, 1))
    hold on
%     [ninf,fpeak] = hinfnorm(sys);
    ninf = norm(sys, inf);
    hinf(i) = ninf;
end

N = 5e5 / length(wc_arr);
crit6 = hinf .* Tmax_norm;
figure()
plot(wc_arr, hinf)
hold on
plot(wc_arr, Tmax_norm)
hold on
plot(wc_arr, crit6)
legend('hinf','Tmax norm', 'crit6')

figure()
plot(wc_arr, crit6)
legend('crit6')

csvwrite(append(path, 'crit6.csv'), [wc_arr', Tmax_norm, hinf, crit6])
