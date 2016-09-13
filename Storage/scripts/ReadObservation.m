#!/usr/bin/octave --silent
# this is a script that is used for making a simple plot
# of the auto and crosscorrelations
# it is not userfriendly, it is only meant to be used on the CDR


if (size(argv)(1) > 0)
  argv
  InputFile = argv(1)
else
  InputFile = "vis.dat";
endif

NoSubbands = 7;
BaseLines = 3; # including autocorrelations
TimeSteps = 60;
Polarisations = 4;
ChannelsPerSubband = 1; # is always 1 for now

# the 2 is because of the 2 floats per complex<float>
datashape = [2, Polarisations, ChannelsPerSubband, BaseLines, TimeSteps, NoSubbands];
NoVis = TimeSteps * NoSubbands * BaseLines * ChannelsPerSubband * \
    Polarisations;

printf("Reading file %s\n", InputFile);
fid = fopen(InputFile, "rb");
rawdata = fread(fid, NoVis * 2, "float");
printf("No Visibilities = %d\n", NoVis)

data = reshape (rawdata, datashape);

c = 1; # there is only 1 channel
if (0==1)
  for t = 1:TimeSteps
    for s = 1:NoSubbands
      for b = 1:BaseLines
	printf("TimeStep %6d - Subband %2d - BaseLine %1d    ", t, s, b);
	printf("POLS (XX, XY, YX, YY): ");
	for p = 1:Polarisations
          printf("(%f, %f)", data(1, p, c, b, t, s), data(2, p, c, b, t, s));
	endfor
	printf("\n")
      endfor
    endfor
  endfor
endif

subband=1;
printf("Plotting subband %d\n", subband);

for t = 1:TimeSteps
  auto1 = data(1, 1, 1, 1, t, subband);
  auto2 = data(1, 1, 1, 3, t, subband);
  crossreal = data(1, 1, 1, 2, t, subband);
  crossimag = data(2, 1, 1, 2, t, subband);
  crosspower = sqrt(crossreal.^2 + crossimag.^2);
  plotdatalog(t,:) = 10* log10([auto1, auto2, abs(crossreal)]);
  plotdatalin(t,:) = [auto1, auto2, crossreal, crossimag];
endfor
# plotdata

#subplot(2,1,1)
#plot(plotdatalin, [';autocor(1x);'; ';autocor(2x);'; ';real(cross(1x,2x));'; ';imag(cross(1x,2x));'])
#title('linear auto and cross correlation versus time')
#xlabel('time (s)')
#ylabel('power')
#subplot(2,1,2)
plot(plotdatalog, [';autocor(1);'; ';autocor(2);'; ';real(cross(1,2));'])
title('auto and cross correlation versus time')
xlabel('time (s)')
ylabel('power (dB)')
pause
