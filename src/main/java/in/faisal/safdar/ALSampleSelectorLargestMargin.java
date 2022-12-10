package in.faisal.safdar;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ALSampleSelectorLargestMargin implements ALSampleSelectionStrategy {
    @Override
    public List<SampleId> selectSamplesForLabeling(ALSampleFeederDataset samples,
                                                   MNISTModel model, List<MNISTModel> auxModels) {
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
                                        (entry1.getValue().largestProbDiff > entry2.getValue().largestProbDiff)
                                                ? 1 : -1
                        ).getKey()
                )
        );
        return l;
    }
}
