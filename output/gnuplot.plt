set terminal svg
set output "output.svg"
set termoption enhanced
set encoding utf8
set datafile separator ' '
set grid

# Set linestyle 1 to blue (#0060ad)
set style line 1 \
    linecolor rgb '#0060ad' \
    linetype 1 linewidth 2 \
    pointtype 7 pointsize 0.5
set style line 2 \
    linecolor rgb '#6000ad' \
    linetype 1 linewidth 2 \
    pointtype 7 pointsize 0.5
set style line 3 \
    linecolor rgb '#ad6000' \
    linetype 1 linewidth 2 \
    pointtype 7 pointsize 0.5
set style line 4 \
    linecolor rgb '#00ad60' \
    linetype 1 linewidth 2 \
    pointtype 7 pointsize 0.5
set style line 5 \
    linecolor rgb '#60ad00' \
    linetype 1 linewidth 2 \
    pointtype 7 pointsize 0.5
set style line 6 \
    linecolor rgb '#ad0060' \
    linetype 1 linewidth 2 \
    pointtype 7 pointsize 0.5

set title "Learning Curves for Active Learning"
set yrange [40:85]
set ylabel "Accuracy"
set xlabel "Active Learning Stages"

plot "UncertaintyLeastConfidence.txt" title "Uncertainty, Least Confidence" \
        with linespoints linestyle 1, \
    "UncertaintyLargestMargin.txt" title "Uncertainty, Largest Margin" \
        with linespoints linestyle 2, \
    "UncertaintySmallestMargin.txt" title "Uncertainty, Smallest Margin" \
        with linespoints linestyle 3, \
    "UncertaintyEntropy.txt" title "Uncertainty, Max Entropy" \
        with linespoints linestyle 4, \
    "QBCVoteEntropy.txt" title "QBC, Max Vote Entropy" \
        with linespoints linestyle 5, \
    "RandomFromPool.txt" title "Random from Pools" \
        with linespoints linestyle 6