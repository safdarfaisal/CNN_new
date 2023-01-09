package in.faisal.safdar;

import org.apache.commons.lang3.tuple.Triple;

import java.awt.*;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ALExperiment {
    ALSampleSelectionStrategy[] experimentStrategies;
    MNISTModel model;
    List<ALMetricsStrategy> listOfMetrics;
    MNISTDataset dataset;
    List<MNISTModel> auxModels;
    int numberOfAuxModels;
    int trainingSampleSize;
    int poolsPerStage;
    int stageCount;
    int stageSize;
    Map <String, ALSampleFeedingStrategy> experimentFeeders;

    private ALExperiment(){

    }

    public ALExperiment create(MNISTDataset dataset, int numberOfAuxModels,
                               int poolsPerStage, int stageCount,
                               int stageSize, int trainingSampleSize, ALSampleSelectionStrategy[] strategies){
        ALExperiment createdExp = new ALExperiment();
        createdExp.numberOfAuxModels = numberOfAuxModels;
        createdExp.experimentStrategies = strategies;
        createdExp.listOfMetrics = new ArrayList<>();
        createdExp.model = new MnistDigitSFCNN();
        createdExp.poolsPerStage = poolsPerStage;
        createdExp.stageCount = stageCount;
        createdExp.stageSize = stageSize;
        createdExp.trainingSampleSize = trainingSampleSize;
        createdExp.experimentFeeders = new HashMap<>();
        createdExp.auxModels = new ArrayList<>();
        createdExp.dataset = dataset;
        return createdExp;
    }

    public void run(){
        //train the first strategy alone.
        ALMetricsStrategy alms = new ALMetricsStrategy(experimentStrategies[0].name(), false);
        ALSampleFeedingStrategy s = ALSampleFeedingStrategy.create(dataset,
                trainingSampleSize, stageCount, stageSize, poolsPerStage,
                experimentStrategies[0], alms);
        listOfMetrics.add(alms);
        experimentFeeders.put(experimentStrategies[0].name(), s);
        //auxModels are created and passed here because we want to use the training cache for QBC.
        //The downside is that auxModels will be trained for stages too for all strategies.
        //TODO: Update the strategy interface to indicate support for auxModels and create and
        //train auxmodels only for strategies that require them.
        IntStream.range(0, numberOfAuxModels).forEach(index -> auxModels.add(new MnistDigitSFCNN()));
        assert s != null;
        s.train(new MnistDigitSFCNN(), auxModels);
        //just clone runner for the remaining strategies.
        Arrays.stream(experimentStrategies).skip(1).forEach(
                strategy -> {
                    ALMetricsStrategy ms = alms.cloneRefurbished(strategy.name());
                    listOfMetrics.add(ms);
                    experimentFeeders.put(
                            strategy.name(),
                            s.cloneRefurbished(dataset, strategy, ms)
                    );
                });
        //run stages for all strategies
        //TODO: these stages can be run in parallel in threads (ensure concurrency)
        IntStream.range(0, stageCount)
                .forEach(i -> experimentFeeders.forEach((j, feeder)
                        -> feeder.runStage(i)));
        //create data files for plotting learning curves
        List<Optional<Image>> l = listOfMetrics.stream().flatMap(
                m -> {
                    m.createOutputForGnuPlot("output");
                    return m.stageLabelsStream();
                }
        ).flatMap(
                t -> t.getRight().orElseGet(ArrayList::new).stream().map(
                        sample -> Triple.of(t.getMiddle(), t.getLeft(), sample)
                )
        ).collect(
                Collectors.groupingBy(
                        Triple::getLeft, Collectors.groupingBy(
                                Triple::getMiddle, Collectors.mapping(
                                        Triple::getRight, Collectors.toList()
                                )
                        )
                )
        ).entrySet().stream().map(
                mel -> ImageDisplay.gridImages(
                        mel.getValue().entrySet().stream().map(
                                el -> ImageDisplay.gridImages(
                                        ImageDisplay.sampleIdsToImages(el.getValue(), dataset),
                                        20, Color.white,
                                        5, el.getKey() + " " + mel.getKey()
                                )
                        ).toList(), 20, Color.white, 7,
                        "Stage: " + mel.getKey()
                )
        ).toList();
        ImageDisplay.gridImages(
                l, 20, Color.white, 1, "Images labeled in stages"
        ).ifPresent(ImageDisplay::show);
        System.out.println("Done");
    }
    public static void main (String[] args){
        ALExperiment exp = new ALExperiment();
        ALSampleSelectionStrategy[] strategies = new ALSampleSelectionStrategy[]{
                new ALSampleSelectorLeastConfidence(),
                new ALSampleSelectorSmallestMargin(),
                new ALSampleSelectorLargestMargin(),
                new ALSampleSelectorEntropy(),
                new ALSampleSelectorQBCVoteEntropy(),
                new ALSampleSelectorQBCKLDivergence(),
                new ALSampleSelectorRandom()
        };
        MNISTDataset ds = new MNISTIDXDataset();
        exp = exp.create(ds, 5, 10,2, 200, 200, strategies);
        System.out.println("Created");
        exp.run();
    }
}
