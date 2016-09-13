#!/bin/bash

# Usage: ./recreateFeedback < *.log
#
# This script (re)generates the Observation12345_feedback files
# from IONProc.log files. Feedback is APPENDED to existing
# Observation*_feedback files.

perl -e '
%feedback_lines = ();

while(<>) {
    # find feedback log lines
    /obs ([0-9]+) .* LTA FEEDBACK: (.*)$/ || next;

    # store the info we wanted
    $obs = $1;
    $feedback = $2;

    # append to feedback file
    open(FEEDBACK, ">>Observation" . $obs . "_feedback");
    print FEEDBACK $feedback . "\n";
    close(FEEDBACK);

    # update logs
    $feedback_lines{$obs} = $feedback_lines{$obs} + 1;
}

# print logs
foreach $obs (keys %feedback_lines)
{
  print "Observation " . $obs . ": found " . $feedback_lines{$obs} . " feedback lines.\n" 
}
'
