package in.faisal.safdar;

import org.apache.commons.lang3.tuple.Triple;

import java.awt.*;
import java.util.List;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

//TODO: Should convert all code to be functional later. Use more exceptions for error cases too.
public class ALSampleFeedingStrategy {
    MNISTDataset ds;
    //TODO: Evaluate and change to a builder model later.
    //TODO: Clean up unused vars.
    int tss;
    int stageCnt;
    int stageSz;
    int poolsPerStage;
    ALSampleSelectionStrategy labelStrategy;
    List<SampleId> trainingSet;
    List<SampleId> testSet;
    List<List<List<SampleId>>> stageValidationPoolSets;
    MNISTModel model;
    ALMetricsStrategy metrics;
    List<MNISTModel> auxModels;

    private ALSampleFeedingStrategy() {

    }

    private ALSampleFeedingStrategy(MNISTDataset dataSet, int trainingSampleSize, int stageCount, int stageSize,
                                    int poolsPerStage, ALSampleSelectionStrategy sampleSelectionStrategy,
                                    ALMetricsStrategy metrics) {
        ds = dataSet;
        tss = trainingSampleSize;
        stageCnt = stageCount;
        //total number of samples processed in a stage, actual count
        //of samples labeled will be a small subset of this
        stageSz = stageSize;
        labelStrategy = sampleSelectionStrategy;
        this.poolsPerStage = poolsPerStage;
        this.metrics = metrics;
    }

    private boolean init() {
        if ((ds == null) || ((tss + stageCnt * stageSz) > ds.trainingSampleCount())) {
            return false;
        }
        /*
        as of now, we are not ensuring that training and validation sample sets/pools are exclusive.
        We are just calling random(), so values might be repeated across the sets. Can be fixed
        later if needed - mostly relevant if we use most of the samples in the original set.
         */
        trainingSet = ds.trainingSubset(tss);
        stageValidationPoolSets = new ArrayList<List<List<SampleId>>>(stageCnt);
        for (int i = 0; i < stageCnt; i++) {
            //stageValidationPoolSets.add(ds.trainingSubset(stageSz));
            List<List<SampleId>> pools = new ArrayList<List<SampleId>>();
            int samplesPerPool = stageSz / poolsPerStage;
            for (int j = 0; j < poolsPerStage; j++) {
                pools.add(ds.trainingSubset(samplesPerPool));
            }
            stageValidationPoolSets.add(pools);
        }
        testSet = ds.testSubset(100);
        return true;
    }

    public MNISTDataset rootDataSource() {
        return ds;
    }

    public MNISTDataset createTrainingDataSource() {
        return ALSampleFeederDataset.create(trainingSet, testSet, ds, trainingSet.size(), testSet.size());
    }

    public List<ALSampleFeederDataset> createALStagePoolDataSources(int stageId) {
        List<List<SampleId>> pools = stageValidationPoolSets.get(stageId);
        List<ALSampleFeederDataset> l = new ArrayList<>(pools.size());
        //training size does not matter in this case, only test would be done with this data source
        pools.forEach(
                (pool) -> {
                    l.add(ALSampleFeederDataset.create(trainingSet, pool, ds, trainingSet.size(), pool.size()));
                }
        );
        return l;
    }

    public void train(MNISTModel m, List<MNISTModel> auxModels) {
        model = m;
        model.init(createTrainingDataSource());
        model.train(trainingSet.size(), false);
        EvalResultMap res = new EvalResultMap();
        model.eval(testSet.size(), res, false);
        Map<String, Object> metricsMap = res.metricsMap;
        metrics.stageMetrics.add(metricsMap);

        if (auxModels != null) {
            this.auxModels = auxModels;
            auxModels.forEach(
                    auxModel -> {
                        auxModel.init(createTrainingDataSource());
                        auxModel.train(trainingSet.size(), false);
                    }
            );
        }
    }

    public ALSampleFeedingStrategy cloneRefurbished(MNISTDataset newDs,
                                                    ALSampleSelectionStrategy newLabelStrategy,
                                                    ALMetricsStrategy newMetrics) {
        /*
        Training and test sets and stage validation pool sets are not deep copied.
        These are considered immutable. Models might have stale datasets after deep
        copy, would have to set correct one when needed before use.
        TODO: Replace all data structures with immutable variants where required.
        Works nicely with move to functional model.
         */
        ALSampleFeedingStrategy clone = new ALSampleFeedingStrategy();
        clone.ds = newDs;
        clone.tss = tss;
        clone.stageCnt = stageCnt;
        clone.stageSz = stageSz;
        clone.poolsPerStage = poolsPerStage;
        clone.labelStrategy = newLabelStrategy;
        clone.trainingSet = trainingSet;
        clone.testSet = testSet;
        clone.stageValidationPoolSets = stageValidationPoolSets;
        clone.model = model.deepCopy();
        clone.metrics = newMetrics;
        clone.auxModels = new ArrayList<>();
        auxModels.forEach(
                auxModel -> {
                    clone.auxModels.add(auxModel.deepCopy());
                }
        );
        return clone;
    }

    public void runStage(int stageId) {
        //sample selection for labeling and training
        List<ALSampleFeederDataset> datasets = createALStagePoolDataSources(stageId);
        List<SampleId> l = new ArrayList<>();
        datasets.forEach(
                (ds) -> {
                    l.addAll(labelStrategy.selectSamplesForLabeling(ds, model, auxModels));
                }
        );
        //training with newly "labeled" samples
        MNISTDataset stageLabeledDS = ALSampleFeederDataset.create(l, testSet, ds, l.size(), testSet.size());
        model.setDataset(stageLabeledDS);
        EvalResultMap res = new EvalResultMap();
        model.train(l.size(), false);
        res = model.eval(testSet.size(), res, false);
        Map<String, Object> metricsMap = res.metricsMap;

        if (metrics.debug && (metrics.name.equals("QBCKLDivergence") || metrics.name.equals("QBCVoteEntropy") ||
                metrics.name.equals("UncertaintySmallestMargin"))) {
            metricsMap.put("StageLabels", l);
        }
        metrics.stageMetrics.add(metricsMap);

        if (auxModels != null) {
            auxModels.forEach(
                    auxModel -> {
                        auxModel.setDataset(ALSampleFeederDataset.create(l, testSet, ds, l.size(), testSet.size()));
                        auxModel.train(l.size(), false);
                    }
            );
        }
    }

    public static ALSampleFeedingStrategy create(MNISTDataset dataSet, int trainingSampleSize, int stageCount,
                                                 int stageSize, int poolsPerStage,
                                                 ALSampleSelectionStrategy sampleSelectionStrategy,
                                                 ALMetricsStrategy metrics) {
        ALSampleFeedingStrategy feeder = new ALSampleFeedingStrategy(dataSet, trainingSampleSize, stageCount,
                stageSize, poolsPerStage, sampleSelectionStrategy, metrics);
        if (feeder.init()) {
            return feeder;
        }
        return null;
    }
}
