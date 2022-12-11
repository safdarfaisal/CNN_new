package in.faisal.safdar;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ALSampleSelectorEntropy implements ALSampleSelectionStrategy {
    @Override
    public String name() {
        return "UncertaintyEntropy";
    }

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
                        Collections.max(
                                m.resultMap.entrySet(),
                                (entry1, entry2) -> (entry1.getValue().entropy > entry2.getValue().entropy)? 1 : -1
                        ).getKey()
                )
        );
        return l;
    }
}
