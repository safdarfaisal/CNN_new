package in.faisal.safdar;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

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

    ALMetricsStrategy cloneRefurbished(String newName) {
        ALMetricsStrategy clone = new ALMetricsStrategy(newName, debug);
        globalMetrics.entrySet().forEach(
                entry -> {
                    clone.globalMetrics.put(entry.getKey(), entry.getValue());
                }
        );
        stageMetrics.forEach(
                map -> {
                    Map<String, Object> m = new HashMap<>();
                    map.entrySet().forEach(
                            entry -> {
                                m.put(entry.getKey(), entry.getValue());
                            }
                    );
                    clone.stageMetrics.add(m);
                }
        );
        return clone;
    }

    void createOutputForGnuPlot(String plotDataFolderPath)
    {
        //filter for each metric later when we have multiple.
        String outFile = plotDataFolderPath + "/" + name + ".txt";
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(outFile));
            int i = 0;
            ListIterator<Map<String, Object>> iter = stageMetrics.listIterator();
            while (iter.hasNext()) {
                Map<String, ?> metric = iter.next();
                writer.write(String.valueOf(i));
                writer.write(" ");
                writer.write(String.valueOf(metric.get("PercentAccuracy")));
                writer.write("\n");
                i++;
            }
            writer.close();
        } catch(IOException e) {
            e.printStackTrace();
        }
    }
}