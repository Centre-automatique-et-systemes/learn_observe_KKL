% Matlab reshapes opposite of python, by columns and not by rows!! 
% Solution: reshape for transpose of what you need then transpose result!
clear all
dx = 2;
dy = 1;
dz = 3;
wc_arr = linspace(0.3, 3, 10);

path = "runs/SaturatedVanDerPol/Supervised_noise/T_star/Paper_Lukas/Test_paper/exp_10_wc0.3-3/zi_mesh_BFsampling1e5uniform/";
Darr = table2array(readtable(append(path, 'D_arr.csv')));
Darr = Darr(:, 2:end);

%%

% zbar = argsup(dT/dz (z_i))
% Tmax = dT/dz (zbar)
% Criterion 1: norm(Tmax) * sup(G(jw))

figure()
hinf = zeros(length(wc_arr), 1);
Tmax_norm = zeros(length(wc_arr), 1);

for i = 1:length(wc_arr)
    wc = wc_arr(i);
    Tmax = table2array(readtable(append(path, 'Tmax_wc', sprintf('%0.2g', wc), '.csv')));
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

csvwrite(append(path, 'wc_arr.csv'), wc_arr.')
csvwrite(append(path, 'crit1.csv'), crit1)

%%

% Criterion 2: sup(Tmax * G(jw))

figure()
hinf = zeros(length(wc_arr), 1);
Tmax_norm = zeros(length(wc_arr), 1);

for i = 1:length(wc_arr)
    wc = wc_arr(i);
    %Tmax = table2array(readtable(append(path, 'Tmax_wc', sprintf('%0.1f', wc), '.csv')));
    %Tmax = table2array(readtable(append(path, 'Tmax_wc', sprintf('%0.2g', wc), '.csv')));
    Tmax = table2array(readtable(append(path, 'Tmax_wc', sprintf('%0.3g', round(wc, 2)), '.csv')));
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
legend('hinf', 'crit2')

figure()
plot(wc_arr, crit2)
legend('crit2')

csvwrite(append(path, 'wc_arr.csv'), wc_arr.')
csvwrite(append(path, 'crit2.csv'), crit2)


%%

% Criterion 3: sup_{z_i, w} (dT/dz (z_i) G(jw))

hinf = zeros(length(wc_arr), 1);

for i = 1:length(wc_arr)
    wc = wc_arr(i);
    %dTdz = table2array(readtable(append(path, 'dTdz_wc', sprintf('%0.1f', wc), '.csv')));
    %dTdz = table2array(readtable(append(path, 'dTdz_wc', sprintf('%0.2g', wc), '.csv')));
    dTdz = table2array(readtable(append(path, 'dTdz_wc', sprintf('%0.3g', round(wc, 2)), '.csv')));
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

csvwrite(append(path, 'wc_arr.csv'), wc_arr.')
csvwrite(append(path, 'crit3.csv'), crit3)

%%

% Criterion 4: norm(dTdz, 2) * sup(G(jw))
% Same as criterion 1 but norm l2 of whole dTdz over all z_i

figure()
hinf = zeros(length(wc_arr), 1);
Tmax_norm = zeros(length(wc_arr), 1);

for i = 1:length(wc_arr)
    wc = wc_arr(i);
    dTdz = table2array(readtable(append(path, 'dTdz_wc', sprintf('%0.3g', round(wc, 2)), '.csv')));
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
    ninf = norm(sys, inf);
    hinf(i) = ninf;
end

N = 5e5 / length(wc_arr);
crit4 = hinf .* Tmax_norm;
figure()
plot(wc_arr, hinf)
hold on
plot(wc_arr, Tmax_norm)
hold on
plot(wc_arr, crit4)
legend('hinf','Tmax norm', 'crit4')

figure()
plot(wc_arr, crit4)
legend('crit4')

csvwrite(append(path, 'wc_arr.csv'), wc_arr.')
csvwrite(append(path, 'crit4.csv'), crit4)
