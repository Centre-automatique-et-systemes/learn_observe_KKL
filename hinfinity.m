dx = 2;
dy = 1;
dz = 3;
hinf = [];
wc_arr = linspace(0.3, 3, 10);

path = "runs/SaturatedVanDerPol/Supervised_noise/T_star/Paper_Lukas/Test_paper/exp_10_wc/zi_mesh_BFsampling1e5uniform/";
Darr = readtable(append(path, 'D_arr.csv'))

% Criterion 1: sup(dTdz(z_i)) * sup(G(jw))
figure()
for i = 1:length(wc_arr)
    wc = wc_arr(i);
    Tmax = table2array(readtable(append(path, 'Tmax_wc', string(wc), '.csv')))
    D = reshape(table2array(Darr(i, :)), [dz, dz]).';
    F = ones(length(D), dy); 
    sys = ss(D, F, eye(length(D)), zeros(length(D), dy));
    bode(sys(1, 1))
    hold on
%     [ninf,fpeak] = hinfnorm(sys);
    ninf = norm(sys, inf);
    hinf(i) = ninf;
end

N = 5e5 / length(wc_arr);
Tmax_norm = vecnorm(table2array(Tmaxarr), 2, 2);
crit1 = hinf.' .* Tmax_norm;
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

% Criterion 2: sup(dTdz_max * G(jw))
figure()
for i = 1:length(wc_arr)
    Tmax = reshape(table2array(Tmaxarr(i,:)), [dz, dx]).';
    D = reshape(table2array(Darr(i, :)), [dz, dz]).';
    F = ones(length(D), dy); 
    sys = ss(D, F, Tmax * eye(length(D)), zeros(size(Tmax, 1), dy));
    bode(sys(1, 1))
    hold on
    ninf = norm(sys, inf);
    hinf(i) = ninf;
end

N = 5e5 / length(wc_arr);
Tmax_norm = vecnorm(table2array(Tmaxarr), 2, 2);
crit2 = hinf;
figure()
plot(wc_arr, hinf)
hold on
plot(wc_arr, Tmax_norm)
hold on
plot(wc_arr, crit2)
legend('hinf','Tmax norm', 'crit2')

figure()
plot(wc_arr, crit2)
legend('crit2')

% Criterion 3: sup(dTdz(z_i) * G(jw))
figure()
for i = 1:length(wc_arr)
    wc = wc_arr(i);
    dTdz = readtable(append(path, 'dTdz_wc', string(wc), '.csv'))
    D = reshape(table2array(Darr(i, :)), [dz, dz]).';
    F = ones(length(D), dy); 
    sys = ss(D, F, dTdz * eye(length(D)), zeros(size(Tmax, 1), dy));
    bode(sys(1, 1))
    hold on
    ninf = norm(sys, inf);
    hinf(i) = ninf;
end

N = 5e5 / length(wc_arr);
Tmax_norm = vecnorm(table2array(Tmaxarr), 2, 2);
crit3 = hinf;
figure()
plot(wc_arr, hinf)
hold on
plot(wc_arr, Tmax_norm)
hold on
plot(wc_arr, crit3)
legend('hinf','Tmax norm', 'crit3')

figure()
plot(wc_arr, crit3)
legend('crit3')
