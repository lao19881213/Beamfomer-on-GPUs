# ******************************************************************************
# Open file
datafile = open('''amplitudes.output''', 'r')
data_raw = []
freq = []

# ******************************************************************************
# Get the raw data, convert header to name list and float list
header_name = []
header_values = []
# convert to python lists
for idx, row in enumerate(datafile):
    # get the header names
    if (idx == 0):
        header_name = row.strip().split(',');
        continue
    # get the values
    if (idx == 1):
        header_values_raw = row.strip().split(',')
        for value in header_values_raw:
            header_values.append(float(value))
        continue

    # Get the data
    split_row = row.strip().split(',')
    data_raw.append(split_row[1:-1]) # skip the frequency id and the last trailing ,
    freq.append(split_row[0])

header = dict(zip(header_name, header_values))
print header

# ******************************************************************************
# Convert raw data to floats
data = []
# convert to float
for row_raw in  data_raw:
    row = []
    for entry in row_raw:
        row.append(float(entry))
    data.append(row)

# plot the content of the data array as a histogram
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


# ******************************************************************************
# Do some visualizations: Plot all available data
from pylab import *

#===============================================================================
axes([0.1, 0.15, 0.8, 0.75])
for idx, output in enumerate(data):
    ax = subplot(7, 7, idx + 1)
    ax.set_yscale('log')
    plot(output)
#===============================================================================

#*******************************************************************************
# Get the values for only freq == 128 and 129
# These should be found in index 128 and 129 : fft == entry starts at 0

freq_begin = []   # 128 
freq_end = []  # 129
freq_rest = []

max_energy = max(max(data))

for row in data:
    energy_begin = row[int(header["freq_begin"])]
    energy_end = row[int(header["freq_end"])]
    freq_begin.append(1.0 * energy_begin / max_energy)
    freq_end.append(1.0 * energy_end / max_energy)

    energy_rest = (sum(row[1:]) - energy_begin - energy_end) / (len(row) - 3)
    freq_rest.append(1.0 * energy_rest / max_energy)


print freq_begin
print freq_end

figure(2)
ax = subplot(1, 1, 1)
ax.set_yscale('log')
p1, = plot(freq_begin, linewidth = 1.0)
p2, = plot(freq_end, linewidth = 1.0)
p3, = plot(freq_rest, linewidth = 1.0)
legend([p1, p2, p3], ["Channel " + str(int(header["freq_begin"])), "Channel " + str(int(header["freq_end"])), "Average others \n minus chan 0"], loc = 7)
title("Energy content of FFT channel " + str(int(header["freq_begin"])) + ", " + str(int(header["freq_end"])) + " and the average of the remaining \n channels for sinussignal of stepwise increasing frequency.")
ylabel("Energy")
xlabel("Input signal")
xlim([0, int(header["freq_steps"]) ])
ylim([10e-8, 2.0])
unformatted_list = arange(float(int(header["freq_begin"])), float(int(header["freq_end"]) + 1), 5 / float(int(header["freq_steps"]) + 1))
locs, labels = xticks(range(0, int(header["freq_steps"]) + 1, 5), [ '%.2f' % elem for elem in unformatted_list ])
plt.setp(labels, rotation = 90)
show()
