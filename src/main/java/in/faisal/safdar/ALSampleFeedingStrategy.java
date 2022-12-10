package in.faisal.safdar;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

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
        model.train(trainingSet.size());
        EvalResultMap res = new EvalResultMap();
        model.eval(testSet.size(), res, false);
        Map<String, Object> metricsMap = res.metricsMap;
        metrics.stageMetrics.add(metricsMap);

        if (auxModels != null) {
            this.auxModels = auxModels;
            auxModels.forEach(
                    auxModel -> {
                        auxModel.init(createTrainingDataSource());
                        auxModel.train(trainingSet.size());
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

    public void eval() {

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
        model.train(l.size());
        res = model.eval(testSet.size(), res, false);
        Map<String, Object> metricsMap = res.metricsMap;
        metrics.stageMetrics.add(metricsMap);

        if (auxModels != null) {
            auxModels.forEach(
                    auxModel -> {
                        auxModel.setDataset(ALSampleFeederDataset.create(l, testSet, ds, l.size(), testSet.size()));
                        auxModel.train(l.size());
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

    public static void main(String[] args) {
        //TODO: Maybe move this to an experiment class later so that multiple experiments can be supported.
        List<ALMetricsStrategy> ml = new ArrayList(2);
        int stageCount = 9;
        ALMetricsStrategy alms = new ALMetricsStrategy("UncertaintyLeastConfidence");
        ALSampleFeedingStrategy s = ALSampleFeedingStrategy.create(new MNISTIDXDataset(),
                1000, stageCount, 1000, 10, new ALSampleSelectorLeastConfidence(),
                alms);
        ml.add(alms);
        //auxModels are created and passed here because we want to use the training cache for QBC.
        //The downside is that auxModels will be trained for stages too for all strategies.
        List<MNISTModel> auxModels = new ArrayList<>();
        for (int j = 0; j < 4; j++) {
            auxModels.add(new MnistDigitSFCNN());
        }
        s.train(new MnistDigitSFCNN(), auxModels);
        //clone runner so that we do not train from scratch every time
        alms = alms.cloneRefurbished("UncertaintySmallestMargin");
        ml.add(alms);
        ALSampleFeedingStrategy s1 =
                s.cloneRefurbished(new MNISTIDXDataset(), new ALSampleSelectorSmallestMargin(), alms);
        alms = alms.cloneRefurbished("UncertaintyLargestMargin");
        ml.add(alms);
        ALSampleFeedingStrategy s2 =
                s.cloneRefurbished(new MNISTIDXDataset(), new ALSampleSelectorLargestMargin(), alms);
        alms = alms.cloneRefurbished("UncertaintyEntropy");
        ml.add(alms);
        ALSampleFeedingStrategy s3 =
                s.cloneRefurbished(new MNISTIDXDataset(), new ALSampleSelectorEntropy(), alms);
        alms = alms.cloneRefurbished("QBCVoteEntropy");
        ml.add(alms);
        ALSampleFeedingStrategy s4 =
                s.cloneRefurbished(new MNISTIDXDataset(), new ALSampleSelectorQBCVoteEntropy(), alms);
        alms = alms.cloneRefurbished("RandomFromPool");
        ml.add(alms);
        ALSampleFeedingStrategy s5 =
                s.cloneRefurbished(new MNISTIDXDataset(), new ALSampleSelectorRandom(), alms);
        for (int i = 0; i < stageCount; i++) {
            //TODO: these stages can be run in parallel in threads (ensure concurrency)
            s.runStage(i);
            s1.runStage(i);
            s2.runStage(i);
            s3.runStage(i);
            s4.runStage(i);
            s5.runStage(i);
        }
        ml.forEach(
                metric -> {
                    metric.createOutputForGnuPlot("output");
                }
        );
        System.out.println("Done");
    }
}
