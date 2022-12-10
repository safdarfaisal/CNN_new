package in.faisal.safdar;

import org.apache.commons.lang3.tuple.Triple;

import java.util.ArrayList;
import java.util.List;

public class ALQBCModelPreparer {
    static List<EvalResultMap> evaluateModel(MNISTModel model,
                                            List<MNISTModel> auxModels, ALSampleFeederDataset samples) {
        EvalResultMap m = new EvalResultMap();
        List<EvalResultMap> results = new ArrayList<>();
        model.setDataset(samples);
        m = model.eval(samples.testSampleCount(), m, false);
        results.add(m);

        auxModels.forEach(
                auxModel -> {
                    EvalResultMap em = new EvalResultMap();
                    auxModel.setDataset(samples.refurbishedClone());
                    em = auxModel.eval(samples.testSampleCount(), em, false);
                    results.add(em);
                }
        );
        return results;
    }
}
