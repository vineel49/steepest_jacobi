% Joint steepest descent and Jacobi based detection method for massive MIMO
% systems
% uncoded QPSK modulation

clear all
close all
clc
Nt = 16; % number of antennas at User
Nr = 128; % number of antennas at Base Station (BS)
fade_var = 0.5; % fade variance of Rayleigh flat channel per dimension
Pav = 2; % average power of QPSK constellation
num_frames = 10^3; % simulation runs

% SNR parameters
SNR_dB = 10; % SNR per bit (dB)
SNR = 10^(0.1*SNR_dB); % SNR in linear scale
noise_var = 2*fade_var*Pav*Nt*Nr/(2*SNR*2*Nt);% noise variance per dimension

tic()
C_Ber = 0; % bit errors initialization
for i1=1:num_frames
% source
a = randi([0 1],1,2*Nt);

% QPSK mapping
seq = 1-2*a(1:2:end)+1i*(1-2*a(2:2:end)); % row vector

% Rayleigh flat-fading channel
fade_chan = normrnd(0,sqrt(fade_var),Nr,Nt)+1i*normrnd(0,sqrt(fade_var),Nr,Nt);

% AWGN
noise = normrnd(0,sqrt(noise_var),Nr,1)+1i*normrnd(0,sqrt(noise_var),Nr,1);

% Channel output
chan_op = fade_chan*seq.' + noise;

%               RECEIVER
% initialization
G = fade_chan'*fade_chan;
A = G + 2*noise_var/Pav*eye(Nt,Nt);
b = fade_chan'*chan_op;
D = diag(diag(A));
Dinv = inv(D);
x0 = Dinv*b;
r0 = b - A*x0;

% first iteration
p0 = A*r0;
u = r0'*r0/(p0'*r0);
x1 = x0 + u*r0 + Dinv*(r0 - u*p0);

% other iterations: Jacobi
K = 5; % number of iterations
for k=2:K
   x1 = Dinv*((D-A)*x1 + b);     
end

dec_seq = x1.'; % decoded sequence now a row vector

% demapping to bits
dec_a = zeros(1,2*Nt);
dec_a(1:2:end) = real(dec_seq)<0;
dec_a(2:2:end) = imag(dec_seq)<0;

% bit errors
C_Ber = C_Ber + nnz(a-dec_a);
end
toc()

% Bit error rate
BER = C_Ber/(num_frames*Nt)