package in.faisal.safdar;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public class ALSampleSelectorRandom implements ALSampleSelectionStrategy {
    @Override
    public String name() {
        return "RandomFromPool";
    }

    public List<SampleId> selectSamplesForLabeling(ALSampleFeederDataset samples,
                                                   MNISTModel model, List<MNISTModel> auxModels) {
        int r = ThreadLocalRandom.current().nextInt(0, samples.testSampleCount());
        List<SampleId> l = new ArrayList<>();
        MNISTImage sample = null;
        for (int i=0; i < r; i++) {
            try {
                sample = samples.testSample();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        if (sample != null) {
            l.add(sample.id());
        }
        return l;
    }
}
