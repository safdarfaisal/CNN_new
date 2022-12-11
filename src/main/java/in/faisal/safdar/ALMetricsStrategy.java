package in.faisal.safdar;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class ALMetricsStrategy {
    String name;
    //Metric maps
    Map<String, Object> globalMetrics;
    //List of metric maps per strategy, ordered by stage id
    List<Map<String, Object>> stageMetrics;
    Boolean debug = false;

    ALMetricsStrategy(String name, Boolean debug) {
        globalMetrics = new HashMap<>();
        stageMetrics = new ArrayList<>();
        this.name = name;
        this.debug = debug;
    }

    public String getName() {
        return name;
    }

    ALMetricsStrategy cloneRefurbished(String newName) {
        ALMetricsStrategy clone = new ALMetricsStrategy(newName, debug);
        clone.globalMetrics.putAll(globalMetrics);
        stageMetrics.forEach(
                map -> {
                    Map<String, Object> m = new HashMap<>(map);
                    clone.stageMetrics.add(m);
                }
        );
        return clone;
    }

    void createOutputForGnuPlot(String plotDataFolderPath) {
        //filter for each metric later when we have multiple.
        String outFile = plotDataFolderPath + "/" + name + ".txt";
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(outFile));
            int i = 0;
            for (Map<String, ?> metric : stageMetrics) {
                writer.write(String.valueOf(i));
                writer.write(" ");
                writer.write(String.valueOf(metric.get("PercentAccuracy")));
                writer.write("\n");
                i++;
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    Stream<Triple<String, Integer, Optional<List<SampleId>>>> stageLabelsStream() {
        return IntStream.range(0, stageMetrics.size()).mapToObj(
                i -> (Triple.of(
                        name, i, Optional.ofNullable((List<SampleId>) (stageMetrics.get(i).get("StageLabels"))))
                )
        );
    }
}