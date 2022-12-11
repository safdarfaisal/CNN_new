The script below is used to generate chart using GnuPlot from VSCode using MPE plugin.

```gnuplot {cmd}
set terminal svg
set termoption enhanced
set encoding utf8
set datafile separator ' '
set grid
set output "output.svg"

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
set style line 7 \
    linecolor rgb '#333333' \
    linetype 1 linewidth 2 \
    pointtype 7 pointsize 0.5

set title "Learning Curves for Active Learning"
set yrange [60:100]
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
        with linespoints linestyle 7
```


```gnuplot {cmd}
set terminal svg
set termoption enhanced
set encoding utf8
set datafile separator ' '
set grid
set output "output1.svg"

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
set style line 7 \
    linecolor rgb '#333333' \
    linetype 1 linewidth 2 \
    pointtype 7 pointsize 0.5

set title "Learning Curves for Active Learning"
set yrange [10:100]
set ylabel "Accuracy"
set xlabel "Active Learning Stages"

plot "UncertaintySmallestMargin.txt" title "Uncertainty, Smallest Margin" \
        with linespoints linestyle 3, \
    "QBCVoteEntropy.txt" title "QBC, Max Vote Entropy" \
        with linespoints linestyle 5, \
    "QBCKLDivergence.txt" title "QBC, Max KL Divergence" \
        with linespoints linestyle 6, \
    "RandomFromPool.txt" title "Random from Pools" \
        with linespoints linestyle 7
```
