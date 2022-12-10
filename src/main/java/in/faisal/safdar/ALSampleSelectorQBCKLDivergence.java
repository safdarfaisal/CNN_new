package in.faisal.safdar;

import org.apache.commons.lang3.tuple.Pair;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static java.util.function.Function.identity;

/*
KLD = sumOf(List<ModelDivergence>)/C
List<ModelDivergence> contains divergences for each model for this sample.
divergence(model) = sumOverClasses(List<modelKldForClass>)
modelKldForClass = modelClassProb * log10(modelClassProb/committeeClassProb)
committeeClassProb = (sumAcrossCommittees(modelClassProb))/C
C is the committee size
*/
public class ALSampleSelectorQBCKLDivergence implements ALSampleSelectionStrategy{
    public List<SampleId> selectSamplesForLabeling(ALSampleFeederDataset samples,
                                                   MNISTModel model, List<MNISTModel> auxModels) {
        class FlatSample {
          int modelId; //0 for main model, the rest for aux models
          SampleId sampleId;
          int classId;
          Double classProb;

          public int classId() {
              return classId;
          }
          public String sampleIdValue() {
              return sampleId.value();
          }
          public Double classProb() {
              return classProb;
          }
          public int modelId() {
              return modelId;
          }
        };
        if (auxModels == null) {
            return null;
        }
        List<EvalResultMap> results = ALQBCModelPreparer.evaluateModel(model, auxModels, samples);
        int committeeSize = auxModels.size() + 1;
        List<FlatSample> flatten = IntStream.range(0, results.size()).mapToObj(j -> {
            EvalResultMap result = results.get(j);
            return result.resultMap.values().stream().map(pr -> Pair.of(pr.index, pr.classProbs)).flatMap(p -> {
                List<Double> l = p.getRight();
                return IntStream.range(0, l.size()).mapToObj(i -> {
                    FlatSample fs = new FlatSample();
                    fs.classId = i;
                    fs.classProb = l.get(i);
                    fs.modelId = j;
                    fs.sampleId = p.getLeft();
                    return fs;
                });
            });
        }).flatMap(identity()).collect(Collectors.toList());

        //Sample->Class->aveClassProbAcrossModels
        Map<String, Map<Integer, Double>> committeeClassProbs =
                flatten.stream().collect(
                        Collectors.groupingBy(
                                FlatSample::sampleIdValue,
                                Collectors.groupingBy(
                                        FlatSample::classId,
                                        Collectors.summingDouble(fs -> fs.classProb/committeeSize)
                                )
                        )
                );

        /*
        TODO: Can do the sum in one step as we have done now - delete this later.
        //Sample->Model->Class->KLDTerm
        Map<String, Map<Integer, Map<Integer, Double>>> modelKldForClasses = flatten.collect(
                Collectors.groupingBy(
                        FlatSample::sampleIdValue,
                        Collectors.groupingBy(
                                FlatSample::modelId,
                                Collectors.groupingBy(
                                        FlatSample::classId,
                                        //only a single value
                                        Collectors.summingDouble(
                                                fs -> fs.classProb*Math.log10(
                                                        fs.classProb/committeeClassProbs.get(
                                                                fs.sampleIdValue()
                                                        ).get(fs.classId)
                                                )
                                        )
                                )
                        )
                )
        );
        //Sample->Model->DivergenceAcrossClasses
        Map<String, Map<Integer, Double>> modelDivergences = modelKldForClasses.entrySet().stream().map(m -> {
            Map<Integer, Double> divergences = m.getValue().entrySet().stream().map(e -> {
                //modelId, divergence
                return Pair.of(e.getKey(), e.getValue().values().stream().mapToDouble(d -> d).sum());
            }).collect(Collectors.toMap(Pair::getLeft, Pair::getRight));
            //sampleId str, divergences
            return Pair.of(m.getKey(), divergences);
        }).collect(Collectors.toMap(Pair::getLeft, Pair::getRight));
        */

        //Sample->KL-Divergence
        Map<String, Double> klDivergences = flatten.stream().collect(
                Collectors.groupingBy(
                        FlatSample::sampleIdValue,
                        Collectors.summingDouble(
                                fs -> fs.classProb*Math.log10(
                                        fs.classProb/committeeClassProbs.get(
                                                fs.sampleIdValue()
                                        ).get(fs.classId)
                                )
                        )
                )
        );
        //using the list to keep the interface consistent though we will have only one value
        //with this strategy until we support selecting multiple entries from the pool.
        List<SampleId> l = new ArrayList<>();
        l.add(
                new SampleId(
                        Collections.max(
                                klDivergences.entrySet(),
                                (entry1, entry2) -> (entry1.getValue() > entry2.getValue())? 1 : -1
                        ).getKey()
                )
        );
        return l;
    }
}
