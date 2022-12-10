package in.faisal.safdar;

import java.util.HashMap;
import java.util.Map;

public class EvalResultMap {
    Map<String, SamplePredictor> resultMap;
    Map<String, Object> metricsMap;

    EvalResultMap() {
        resultMap = new HashMap();
        metricsMap = new HashMap();
    }
}
