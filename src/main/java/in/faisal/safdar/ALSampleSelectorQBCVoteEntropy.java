package in.faisal.safdar;

import java.util.*;

public class ALSampleSelectorQBCVoteEntropy implements ALSampleSelectionStrategy {
    public List<SampleId> selectSamplesForLabeling(ALSampleFeederDataset samples,
                                                   MNISTModel model, List<MNISTModel> auxModels) {
        if (auxModels == null) {
            return null;
        }
        List<EvalResultMap> results = ALQBCModelPreparer.evaluateModel(model, auxModels, samples);
        int committeeSize = auxModels.size() + 1;
        Map<String, List<Integer>> votes = new HashMap<String, List<Integer>>();
        //add a vote list for all samples. All results have the same samples, so just one is enough.
        results.get(0).resultMap.keySet().forEach(
                key -> {
                    List<Integer> l = new ArrayList<Integer>(10);
                    for (int i=0; i < 10; i++) {
                        l.add(0);
                    }
                    votes.put(key, l);

                }
        );
        results.forEach(
                //walk through each entry and put the votes in the right place.
                result -> result.resultMap.entrySet().forEach(
                        entry -> {
                            SamplePredictor pr = entry.getValue();
                            List<Integer> labelVotes = votes.get(pr.index.value());
                            if (labelVotes == null){
                                System.out.println(pr.index.value());
                            }
                            int i = labelVotes.get(pr.predictedLabel);
                            i++;
                            labelVotes.add(pr.predictedLabel, i);
                        }
                )
        );
        Map<String, Float> entropies = new HashMap<String, Float>();
        votes.entrySet().forEach(
                (entry) -> {
                    String sampleId = entry.getKey();
                    List<Integer> sampleVotes = entry.getValue();
                    ListIterator<Integer> iter = sampleVotes.listIterator();
                    Double entropy = 0.0d;
                    while (iter.hasNext()) {
                        int v = iter.next();
                        if (v != 0) {
                            entropy = entropy - (v/committeeSize)*Math.log10(v/committeeSize);
                        }
                    }
                    entropies.put(sampleId, entropy.floatValue());
                }
        );
        //using the list to keep the interface consistent though we will have only one value
        //with this strategy until we support selecting multiple entries from the pool.
        List<SampleId> l = new ArrayList<>();
        l.add(
                new SampleId(
                        Collections.max(
                                entropies.entrySet(),
                                (entry1, entry2) -> (entry1.getValue() > entry2.getValue())? 1 : -1
                        ).getKey()
                )
        );
        return l;
    }
}
