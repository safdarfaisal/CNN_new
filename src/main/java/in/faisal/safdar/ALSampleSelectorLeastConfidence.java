package in.faisal.safdar;

import java.util.*;

public class ALSampleSelectorLeastConfidence implements ALSampleSelectionStrategy {
    public List<SampleId> selectSamplesForLabeling(ALSampleFeederDataset samples,
                                                   MNISTModel model, List<MNISTModel> auxModels) {
        //Map<SampleId, SamplePredictor> m = new HashMap();
        EvalResultMap m = new EvalResultMap();
        model.setDataset(samples);
        m = model.eval(samples.testSampleCount(), m, false);
        //using the list to keep the interface consistent though we will have only one value
        //with this strategy until we support selecting multiple entries from the pool.
        List<SampleId> l = new ArrayList<>();
        l.add(
                new SampleId(
                        Collections.min(
                                m.resultMap.entrySet(),
                                (entry1, entry2) ->
                                        (entry1.getValue().probForPrediction > entry2.getValue().probForPrediction)
                                                ? 1 : -1
                        ).getKey()
                )
        );
        return l;
    }
}
