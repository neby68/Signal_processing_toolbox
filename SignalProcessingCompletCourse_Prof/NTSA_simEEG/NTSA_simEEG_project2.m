%%
%     COURSE: Solved challenges in neural time series analysis
%    SECTION: Simulating EEG data
%      VIDEO: Project 2: dipole-level EEG data
% Instructor: sincxpress.com
%
%%

%% 

% mat file containing EEG, leadfield and channel locations
load emptyEEG

% select dipole location (more-or-less random)
diploc = 25;

% plot brain dipoles
figure(), clf, subplot(121)
plot3(lf.GridLoc(:,1), lf.GridLoc(:,2), lf.GridLoc(:,3), 'bo','markerfacecolor','y')
hold on
plot3(lf.GridLoc(diploc,1), lf.GridLoc(diploc,2), lf.GridLoc(diploc,3), 'rs','markerfacecolor','k','markersize',10)
rotate3d on, axis square
title('Brain dipole locations')


% Each dipole can be projected onto the scalp using the forward model. 
% The code below shows this projection from one dipole.
subplot(122)
topoplotIndie(-lf.Gain(:,1,diploc), EEG.chanlocs,'numcontour',0,'electrodes','numbers','shading','interp');
set(gca,'clim',[-1 1]*40)
title('Signal dipole projection')

%% add signal to one dipole and project to scalp

% reduce data size a bit
EEG.pnts  = 2000;
EEG.times = (0:EEG.pnts-1)/EEG.srate;

% initialize all dipole data
dipole_data = zeros(size(lf.Gain,3),EEG.pnts);

% add signal to one dipole
dipole_data(diploc,:) = sin(2*pi*10*EEG.times);

% now project dipole data to scalp electrodes
EEG.data = squeeze(lf.Gain(:,1,:))*dipole_data;

% plot the data
plot_simEEG(EEG,31,2);

%% now for the projects!

%%%% IMPORTANT! Check internal consistency with existing EEG structure!!!

EEG


%% 1) pure sine wave with amplitude explorations

EEG.trials = 40;

% dipole amplitude magnitude
ampl = 1*10^-10;
freq = 10;

% initialize all dipole data
dipole_data = zeros(size(lf.Gain, 3), EEG.pnts);

% compute one dipole
dipole_data(diploc, :) = ampl*sin(2*pi*freq*EEG.times);
signal = squeeze(lf.Gain(:,1,:))*dipole_data;

% repeat that for N trials
EEG.data = zeros(EEG.nbchan, EEG.pnts, EEG.trials)
for trial=1:EEG.trials
    EEG.data(:,:,trial) = signal;
end

% plot the data
plot_simEEG(EEG,31,2);




%%% Question: What is the smallest amplitude of dipole signal that still
%             elicits a scalp-level response?

% the amplitude on time either time and frequency domain will become
% smaller....it probably depends on the electrode specification ?
%% 2) sine wave with noise

%%% Question: Given amplitude=1 of dipole signal, what standard deviation of noise
%             at all other dipoles overpowers the signal (qualitatively)?

EEG.trials = 40;
% dipole amplitude magnitude
ampl = 1;
freq = 10;
noise_ampl = 1e0;

% repeat that for N trials
EEG.data = zeros(EEG.nbchan, EEG.pnts, EEG.trials)
for trial=1:EEG.trials
    % initialize all dipole data
    % do it for all the trial, because each trial has its own noise
    % caracteristics
    dipole_data = noise_ampl*randn(size(lf.Gain, 3), EEG.pnts);
    
    % compute one dipole
    dipole_data(diploc, :) = ampl*sin(2*pi*freq*EEG.times);
    EEG.data(:,:,trial)  = squeeze(lf.Gain(:,1,:))*dipole_data;
end

% plot the data
plot_simEEG(EEG,31,2);



%% 3) Non-oscillatory transient in one dipole, noise in all other dipoles

EEG.trials = 40;

% dipole amplitude magnitude
noise_ampl = 1;

% Gaussian
peaktime = 1; % seconds
width    = .12;
ampl     = 70;
gaus = ampl * exp( -(EEG.times-peaktime).^2 / (2*width^2) );

% figure()
% plot(gaus)


% repeat that for N trials
EEG.data = zeros(EEG.nbchan, EEG.pnts, EEG.trials)
for trial=1:EEG.trials
    % initialize all dipole data
    % do it for all the trial, because each trial has its own noise
    % caracteristics
    dipole_data = noise_ampl*randn(size(lf.Gain, 3), EEG.pnts);
    
    % compute one dipole
    dipole_data(diploc, :) = gaus;
    EEG.data(:,:,trial)  = squeeze(lf.Gain(:,1,:))*dipole_data;
end

% plot the data
plot_simEEG(EEG,31,2);

%% 4) Non-stationary oscillation in one dipole, transient oscillation in another dipole, noise in all dipoles

%%% first pick two dipoles
dipole1 = 109;
dipole2 = 510;
%%% then do the simulation
EEG.trials = 40;

% dipole amplitude magnitude
noise_ampl = 1;

% Gaussian
peaktime = 1; % seconds
width    = .12;
ampl     = 70;
gaus = ampl * exp( -(EEG.times-peaktime).^2 / (2*width^2) );
sinewave = 1*sin(2*pi*EEG.times*freq)
transient_oscillation = sinewave.*gaus

nb_freq = 10;
f_max = 50;
f_offset = 5;
% plot(rand(1,EEG.pnts), '--','b')
% plot(signal)


% repeat that for N trials
EEG.data = zeros(EEG.nbchan, EEG.pnts, EEG.trials)
for trial=1:EEG.trials
    % initialize all dipole data
    % do it for all the trial, because each trial has its own noise
    % caracteristics
    dipole_data = noise_ampl*randn(size(lf.Gain, 3), EEG.pnts);
    
    fmod = f_offset + (f_max*interp1(rand(1,nb_freq), linspace(0, nb_freq, EEG.pnts)));
    % compute one dipole
    dipole_data(dipole1, :) = transient_oscillation;
    dipole_data(dipole2, :) = sin( 2*pi * ((EEG.times + cumsum(fmod))/EEG.srate) );
    EEG.data(:,:,trial)  = squeeze(lf.Gain(:,1,:))*dipole_data;
end


% plot the data
plot_simEEG(EEG,56,3);
plot_simEEG(EEG,31,2);

%% 
