% QuaDRiGa CSI dataset generation with NR-like DMRS patterns
% Compatible with the ORIGINAL Dataset_Pro (no Python changes):
%   - Saves tensors in shape: [v, n, L, k, a, b, c]
%     where v = speed index, n = UE/sample index per speed
%   - Dataset_Pro splits TRAIN/VAL internally along n with train_per/valid_per.
%
% This "A)" version uses:
%   - Train+Val pool: V_train = 900 speeds (10.1:0.1:100), N_train = 10 UEs each => 9000 total
%     -> with train_per=0.9: train=8100, val=900 (internal to Dataset_Pro)
%   - Test: V_test = 10 speeds (10:10:100), N_test = 1000 samples per speed => 10,000 total
%
% Folder layout:
%   dmrs_datasets/PATTERN_CONFIG/TDD_OR_FDD_CONFIG/
%
% Files saved (keys inside match Dataset_Pro expectations):
%   H_U_his_train.mat  (var: H_U_his_train)
%   H_U_pre_train.mat  (var: H_U_pre_train)
%   H_D_pre_train.mat  (var: H_D_pre_train)
%   H_U_his_test.mat   (var: H_U_his_test)
%   H_U_pre_test.mat   (var: H_U_pre_test)
%   H_D_pre_test.mat   (var: H_D_pre_test)
%   meta.mat

clc; clear; close all;

%% =========================
% CONFIG
%% =========================
cfg.pattern_id = 3;          % 1,2,3
cfg.duplex     = 'FDD';      % 'TDD' or 'FDD'
cfg.out_root   = './dmrs_datasets';

% Train+Val pool (Dataset_Pro splits internally along n)
cfg.train_speeds = 10.1:0.1:100;   % 900 speeds
cfg.N_per_speed_train = 10;        % 10 samples per speed -> 9000 total

% Paper-style test: 10 speeds x 1000 samples/speed -> 10k total
cfg.test_speeds = 10:10:100;       % 10 speeds
cfg.N_test_per_speed = 1000;

% Repro for geometry randomness
cfg.seed_geom = 123;

% Timing/frequency grid (same as original)
cfg.dt         = 0.5e-3;
cfg.Timelength = 19*cfg.dt;
cfg.SnapNum    = 1 + floor(cfg.Timelength/cfg.dt); % 20

cfg.Ktot = 96;
cfg.Kul  = 48;
cfg.Kdl  = 48;

cfg.P_hist = 16;
cfg.L_fut  = 4;
assert(cfg.SnapNum == cfg.P_hist + cfg.L_fut, 'SnapNum must equal P_hist+L_fut.');

% For test generation, generate in chunks for speed
cfg.UENum_per_call = 10;  % how many UEs to generate per QuaDRiGa call (test loops)

%% =========================
% DMRS PATTERN TOGGLE (RB x snapshot abstraction)
%% =========================
switch cfg.pattern_id
    case 1
        cfg.dmrs_time_step = 2;   % every 1 ms (2 snapshots)
        cfg.dmrs_freq_step = 2;   % every 2 RBs
        cfg.pattern_name   = 'pattern1_type1_sparse';
    case 2
        cfg.dmrs_time_step = 2;   % every 1 ms
        cfg.dmrs_freq_step = 1;   % every RB
        cfg.pattern_name   = 'pattern2_type2_densefreq';
    case 3
        cfg.dmrs_time_step = 1;   % every snapshot
        cfg.dmrs_freq_step = 1;   % every RB
        cfg.pattern_name   = 'pattern3_highmob_dense';
    otherwise
        error('Unknown cfg.pattern_id. Use 1,2,3.');
end

cfg.duplex = upper(cfg.duplex);
assert(ismember(cfg.duplex, {'TDD','FDD'}), 'cfg.duplex must be TDD or FDD.');

%% =========================
% Output folder
%% =========================
out_dir = fullfile(cfg.out_root, cfg.pattern_name, cfg.duplex);
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

%% =========================
% QuaDRiGa base parameters / antennas
%% =========================
rng(cfg.seed_geom);

s = qd_simulation_parameters;
s.center_frequency = 2.4e9;

M_BS = 4; N_BS = 4;
Mg_BS = 1; Ng_BS = 1;
ElcTltAgl_BS = 7;
Hspc_Tx_BS = 0.5*s.wavelength;
Vspc_Tx_BS = 0.5*s.wavelength;

BSAntArray = qd_arrayant.generate('3gpp-mmw', M_BS, N_BS, ...
    s.center_frequency, 2, ElcTltAgl_BS, ...
    Vspc_Tx_BS/s.wavelength, Mg_BS, Ng_BS, ...
    Vspc_Tx_BS/s.wavelength*M_BS, Hspc_Tx_BS/s.wavelength*N_BS);

UEAntArray = qd_arrayant.generate('3gpp-mmw', 1, 1, ...
    s.center_frequency, 1, ElcTltAgl_BS, ...
    Vspc_Tx_BS/s.wavelength, Mg_BS, Ng_BS, ...
    Vspc_Tx_BS/s.wavelength*M_BS, Hspc_Tx_BS/s.wavelength*N_BS);

% Frequency-bin indices (interpretation for duplex)
ul_bins = 1:cfg.Kul;
if strcmp(cfg.duplex, 'FDD')
    dl_bins = (cfg.Kul+1):cfg.Ktot;   % 49..96 (pseudo-FDD as in original demo)
else
    dl_bins = 1:cfg.Kul;              % TDD: same band
end

%% =========================
% DMRS masks (same for all samples)
%% =========================
dmrs_ul_rb_idx   = 1:cfg.dmrs_freq_step:cfg.Kul;
dmrs_hist_t_idx  = 1:cfg.dmrs_time_step:cfg.P_hist;

dmrs_fut_abs = (cfg.P_hist+1):cfg.dmrs_time_step:cfg.SnapNum; % subset of 17..20
dmrs_fut_rel = dmrs_fut_abs - cfg.P_hist;                     % subset of 1..4

M_hist_ul = false(cfg.P_hist, cfg.Kul);
M_fut_ul  = false(cfg.L_fut,  cfg.Kul);
M_fut_dl  = false(cfg.L_fut,  cfg.Kdl);

M_hist_ul(dmrs_hist_t_idx, dmrs_ul_rb_idx) = true;
M_fut_ul(dmrs_fut_rel, dmrs_ul_rb_idx) = true;
M_fut_dl(dmrs_fut_rel, dmrs_ul_rb_idx) = true; % stored on 48-bin grid

% Broadcast masks across [N, T, K, a, b, c]
M_hist_ul6 = reshape(single(M_hist_ul), [1, cfg.P_hist, cfg.Kul, 1, 1, 1]);
M_fut_ul6  = reshape(single(M_fut_ul),  [1, cfg.L_fut,  cfg.Kul, 1, 1, 1]);
M_fut_dl6  = reshape(single(M_fut_dl),  [1, cfg.L_fut,  cfg.Kdl, 1, 1, 1]);

%% =========================
% Helper: generate one batch for a given speed (returns UENum samples)
%% =========================
function [his_ul, fut_ul, fut_dl] = gen_batch_for_speed( ...
        speed_kmh, cfg_local, BSAntArray, UEAntArray, ul_bins, dl_bins, ...
        M_hist_ul6, M_fut_ul6, M_fut_dl6, M_BS, N_BS)

    UENum = cfg_local.UENum_per_call;

    % QuaDRiGa parameters for mobility
    s1 = qd_simulation_parameters;
    s1.center_frequency = 2.4e9;
    s1.set_speed(speed_kmh, cfg_local.dt);
    s1.use_random_initial_phase = true;
    s1.use_3GPP_baseline = 1;

    UETrackLength = (speed_kmh/3.6) * cfg_local.Timelength;

    % BS & UE geometry
    BSlocation = [0;0;30];
    rho_min = 20; rho_max = 50;
    rho = rho_min + (rho_max-rho_min) * rand(1, UENum);
    phi = 120*rand(1, UENum) - 60;
    UEcenter = [200;0;1.5];

    UElocation = zeros(3, UENum);
    UEtrack = repmat(qd_track, 1, UENum);

    for ind_UE = 1:UENum
        rho_n = rho(ind_UE);
        phi_n = phi(ind_UE);
        UElocation(:, ind_UE) = [-rho_n*cosd(phi_n); rho_n*sind(phi_n); 0] + UEcenter;

        UEtrack(1, ind_UE) = qd_track.generate('linear', UETrackLength);
        UEtrack(1, ind_UE).name = num2str(ind_UE);
        UEtrack(1, ind_UE).interpolate('distance', 1/s1.samples_per_meter, [], [], 1);
    end

    % Layout
    l1 = qd_layout(s1);
    l1.no_tx = 1;
    l1.tx_array = BSAntArray;
    l1.tx_position = BSlocation;

    l1.no_rx = UENum;
    l1.rx_array = UEAntArray;
    l1.rx_track = UEtrack;
    l1.rx_position = UElocation;
    l1.set_scenario('3GPP_38.901_UMa_NLOS');

    [BS2UE_channel, ~] = l1.get_channels();

    % Prealloc outputs (UENum samples)
    his_ul = zeros(UENum, cfg_local.P_hist, cfg_local.Kul, M_BS, N_BS, 2, 'single');
    fut_ul = zeros(UENum, cfg_local.L_fut,  cfg_local.Kul, M_BS, N_BS, 2, 'single');
    fut_dl = zeros(UENum, cfg_local.L_fut,  cfg_local.Kdl, M_BS, N_BS, 2, 'single');

    for ii = 1:UENum
        h = BS2UE_channel(ii).fr(17280e3, cfg_local.Ktot);
        h = reshape(h, 2, M_BS, N_BS, cfg_local.Ktot, cfg_local.SnapNum);
        h = permute(h, [5,4,3,2,1]); % [T,K,N,M,pol]

        H_his  = single(h(1:cfg_local.P_hist, ul_bins, :, :, :));                 % [16,48,4,4,2]
        H_futU = single(h(cfg_local.P_hist+1:cfg_local.SnapNum, ul_bins, :, :, :)); % [4,48,4,4,2]
        H_futD = single(h(cfg_local.P_hist+1:cfg_local.SnapNum, dl_bins, :, :, :)); % [4,48,4,4,2]

        % Apply DMRS masks (broadcast over antennas/pol)
        H_his  = H_his  .* squeeze(M_hist_ul6(1,:,:,:,:,:));
        H_futU = H_futU .* squeeze(M_fut_ul6(1,:,:,:,:,:));
        H_futD = H_futD .* squeeze(M_fut_dl6(1,:,:,:,:,:));

        his_ul(ii,:,:,:,:,:) = H_his;
        fut_ul(ii,:,:,:,:,:) = H_futU;
        fut_dl(ii,:,:,:,:,:) = H_futD;
    end
end

%% =========================
% Allocate output arrays in Dataset_Pro format: [v, n, L, k, a, b, c]
%% =========================
V_train = numel(cfg.train_speeds);
N_train = cfg.N_per_speed_train;

V_test  = numel(cfg.test_speeds);
N_test  = cfg.N_test_per_speed;

H_U_his_train = zeros(V_train, N_train, cfg.P_hist, cfg.Kul, M_BS, N_BS, 2, 'single');
H_U_pre_train = zeros(V_train, N_train, cfg.L_fut,  cfg.Kul, M_BS, N_BS, 2, 'single');
H_D_pre_train = zeros(V_train, N_train, cfg.L_fut,  cfg.Kdl, M_BS, N_BS, 2, 'single');

H_U_his_test  = zeros(V_test,  N_test,  cfg.P_hist, cfg.Kul, M_BS, N_BS, 2, 'single');
H_U_pre_test  = zeros(V_test,  N_test,  cfg.L_fut,  cfg.Kul, M_BS, N_BS, 2, 'single');
H_D_pre_test  = zeros(V_test,  N_test,  cfg.L_fut,  cfg.Kdl, M_BS, N_BS, 2, 'single');

%% =========================
% Generate TRAIN+VAL pool: one batch per speed, N=10 per speed
%% =========================
fprintf('Generating TRAIN+VAL pool: V=%d speeds, N=%d samples/speed (%d total)\n', ...
    V_train, N_train, V_train*N_train);

for vi = 1:V_train
    sp = cfg.train_speeds(vi);

    cfg_local = cfg;
    cfg_local.UENum_per_call = N_train;  % generate exactly N=10 in one call

    [his_ul, fut_ul, fut_dl] = gen_batch_for_speed( ...
        sp, cfg_local, BSAntArray, UEAntArray, ul_bins, dl_bins, ...
        M_hist_ul6, M_fut_ul6, M_fut_dl6, M_BS, N_BS);

    H_U_his_train(vi,:,:,:,:,:,:) = his_ul;
    H_U_pre_train(vi,:,:,:,:,:,:) = fut_ul;
    H_D_pre_train(vi,:,:,:,:,:,:) = fut_dl;

    if mod(vi, 50) == 0
        fprintf('  train pool progress: %d / %d speeds\n', vi, V_train);
    end
end

%% =========================
% Generate TEST: fixed speeds, N=1000 samples per speed (chunked)
%% =========================
fprintf('Generating TEST: V=%d speeds, N=%d samples/speed (%d total)\n', ...
    V_test, N_test, V_test*N_test);

for vi = 1:V_test
    sp = cfg.test_speeds(vi);

    cfg_local = cfg;
    cfg_local.UENum_per_call = cfg.UENum_per_call; % chunk size (e.g., 10)

    n_filled = 0;
    while n_filled < N_test
        [his_ul, fut_ul, fut_dl] = gen_batch_for_speed( ...
            sp, cfg_local, BSAntArray, UEAntArray, ul_bins, dl_bins, ...
            M_hist_ul6, M_fut_ul6, M_fut_dl6, M_BS, N_BS);

        take = min(cfg_local.UENum_per_call, N_test - n_filled);

        H_U_his_test(vi, n_filled+1:n_filled+take, :, :, :, :, :) = his_ul(1:take,:,:,:,:,:);
        H_U_pre_test(vi, n_filled+1:n_filled+take, :, :, :, :, :) = fut_ul(1:take,:,:,:,:,:);
        H_D_pre_test(vi, n_filled+1:n_filled+take, :, :, :, :, :) = fut_dl(1:take,:,:,:,:,:);

        n_filled = n_filled + take;

        if mod(n_filled, 200) == 0
            fprintf('  test speed %.1f km/h: %d / %d\n', sp, n_filled, N_test);
        end
    end
end

%% =========================
% Save (NO separate val files; Dataset_Pro splits internally along n)
%% =========================
save(fullfile(out_dir, 'H_U_his_train.mat'), 'H_U_his_train', '-v7.3');
save(fullfile(out_dir, 'H_U_pre_train.mat'), 'H_U_pre_train', '-v7.3');
save(fullfile(out_dir, 'H_D_pre_train.mat'), 'H_D_pre_train', '-v7.3');

save(fullfile(out_dir, 'H_U_his_test.mat'), 'H_U_his_test', '-v7.3');
save(fullfile(out_dir, 'H_U_pre_test.mat'), 'H_U_pre_test', '-v7.3');
save(fullfile(out_dir, 'H_D_pre_test.mat'), 'H_D_pre_test', '-v7.3');

meta.cfg = cfg;
meta.M_hist_ul = M_hist_ul;
meta.M_fut_ul  = M_fut_ul;
meta.M_fut_dl  = M_fut_dl;
meta.ul_bins = ul_bins;
meta.dl_bins = dl_bins;
meta.train_speeds = cfg.train_speeds;
meta.test_speeds  = cfg.test_speeds;

save(fullfile(out_dir, 'meta.mat'), 'meta', '-v7.3');

disp(['Saved dataset to: ' out_dir]);

% Notes:
% - For TDD training (u2d=0), your Python should point train_tgt to H_U_pre_train.mat
% - For FDD training (u2d=1), point train_tgt to H_D_pre_train.mat
% - Dataset_Pro will internally use first 90% of n=10 (i.e., 9) for train and last 10% (i.e., 1) for val.