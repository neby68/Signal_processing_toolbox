%%
%     COURSE: Solved challenges in neural time series analysis
%    SECTION: Simulating EEG data
%      VIDEO: Project 1: Channel-level EEG data
% Instructor: sincxpress.com
%
%%

%%% INSTRUCTIONS:
% The goal of this assignment is to simulate time series data
% that can be used to test time-series analysis methods.
% For each section below: 
%   1) Complete the MATLAB code
%   2) Put the data into the EEG structure
%      - Make sure all relevant fields are accurate (EEG.data, EEG.pnts, EEG.trials, EEG.srate, EEG.nbchan, EEG.times)
%   3) Use function plot_simEEG to plot some data

%% 1) pure phase-locked sine wave

% parameters
EEG.srate  = 500; % sampling rate in Hz
EEG.pnts   = 1500;
EEG.trials = 30;
EEG.nbchan = 23;

sinefreq = 6.75; % in Hz

% time vector
EEG.times = (0:EEG.pnts-1)/EEG.srate;


% loop over channels and create data
for chani=1:EEG.nbchan
    for triali=1:EEG.trials
        % data as a sine wave
        EEG.data(chani,:,triali) = sin(2*pi*sinefreq*EEG.times);
    end
end


% plot an ERP from one channel
figure(1), clf
plot(EEG.times,squeeze(mean(EEG.data(10,:,:),3)),'linew',2)
xlabel('Time (s)'), ylabel('Activity')
title('ERP from channel 10')


% the function below takes at least one argument (EEG),
% and optionally a second argument (channel number),
% and optionally a third argument (figure number)
plot_simEEG(EEG,2,3)

%% 2) Non-phase-locked sine wave

% hint: copy/paste the code above but add something inside the sine 
%       function on each trial.

% parameters
EEG.srate  = 500; % sampling rate in Hz
EEG.pnts   = 1500;
EEG.trials = 30;
EEG.nbchan = 23;

sinefreq = 6.75; % in Hz

% time vector
EEG.times = (0:EEG.pnts-1)/EEG.srate;


% loop over channels and create data
for chani=1:EEG.nbchan
    for triali=1:EEG.trials
        % data as a sine wave
        EEG.data(chani,:,triali) = sin(2*pi*sinefreq*EEG.times + 2*pi*rand);
    end
end


% plot an ERP from one channel
figure(), clf
hold on;
for triali=1:1 %EEG.trials
    plot(EEG.times,EEG.data(10,:,triali), 'Color', [0.8 0.8 0.8])
end
plot(EEG.times,squeeze(mean(EEG.data(10,:,:),3)),'linew',2)
xlabel('Time (s)'), ylabel('Activity')
title('ERP from channel 10')
hold off;

% the function below takes at least one argument (EEG),
% and optionally a second argument (channel number),
% and optionally a third argument (figure number)
plot_simEEG(EEG,2)


%% 3) multisine waves

% list of frequencies and corresponding amplitudes
frex = [ 3 5 16 ];
amps = [ 2 4 5  ];


% loop over channels and trials
for chani=1:EEG.nbchan
    for triali=1:EEG.trials
        
        % data as a sine wave plus noise
        EEG.data(chani,:,triali) = 0;
        for freqi=1:length(frex)
            fprintf('Value of freqi: %d\n', freqi);
            EEG.data(chani,:,triali) = EEG.data(chani,:,triali) ...
                                    + amps(freqi)*sin(2*pi*frex(freqi)*EEG.times + 2*pi*rand);
        end
    end
end

figure(), clf
hold on;
for triali=1:2 %EEG.trials
    plot(EEG.times,EEG.data(10,:,triali), 'Color', [0.8 0.8 0.8])
end

plot_simEEG(EEG,2)

%%% Question: What can you change in the code above to make the EEG
%             activity non-phase-locked over trials?

% as section 2 + 2*pi*rand 

%             
%%% Question: Which of the plots look different for phase-locked vs. non-phase-locked?
%             (Hint: plot them in different figures to facilitate comparison.)
%             Are you surprised about the differences?

%ERP plot change the most, but the main frequyency are still the same
%% 4) nonstationary sine waves

% hint: instantaneous frequency via interpolated random numbers
freqmod = 20*interp1(rand(1,10), linspace(1,10,EEG.pnts));
signal  = sin( 2*pi * ((EEG.times + cumsum(freqmod))/EEG.srate) );

figure(), clf
subplot(2, 1, 1);
plot(freqmod)
subplot(2, 1, 2);
plot(signal)    

%% 5) transient oscillations w/ Gaussian


peaktime = 1; % seconds
width = .12;
sinewave = sin(2*pi*EEG.times*sinefreq)

gaus = exp( -(EEG.times-peaktime).^2 / (2*width^2) );

% then multiply the gaussian by a sine wave
signal = sinewave.*gaus;

figure(), clf
% plot(EEG.times, gaus)
plot(signal)

%% 6) repeat #3 with white noise

white_noise = rand(1,EEG.pnts);
signal = sinewave.*white_noise;

figure()
plot(rand(1,EEG.pnts))
figure()
plot(signal)

%% 7) repeat #5 with 1/f noise







%%

