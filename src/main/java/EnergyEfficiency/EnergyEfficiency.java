package EnergyEfficiency;

/*
 * Before you start, you should:
 * 1. Put the CSV file into your resources folder

 * You are an engineer and have collected a dataset on how different shape of buildings affect the energy efficiency.
 * You have to perform an analysis to determine the heating load and cooling load of the building, given a set
 * of attributes which correspond to a shape. Here, the hints of what you should do is given below:

 * Step 1: Read the CSV file in a RecordReader
 * Step 2: Specify the transform process
 * Step 3: Put each instances into the List<List<Writable>>
 * Step 4: Perform shuffling of dataset with seed number provided
 * Step 5: Do train and test splitting
 * Step 6: Do data normalisation in the form of [0,1]
 * Step 7: Define the specifications for early stopping, and subsequently perform early stopping. Your model should not
 *         take more than 200 epochs for training.
 * Step 8: Perform evaluation on both train and test set. Kindly ensure R2 score of outputs on both train and test
 *         is more than +90%.

 * Dataset origin: https://archive.ics.uci.edu/ml/datasets/Energy+efficiency#
 * Dataset attribute description:
 * X1: Relative Compactness
 * X2: Surface Area
 * X3: Wall Area
 * X4: Roof Area
 * X5: Overall Height
 * X6: Orientation
 * X7: Glazing Area
 * X8: Glazing Area Distribution
 * Y1: Heating Load
 * Y2: Cooling Load
 */

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.filter.FilterInvalidValues;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class EnergyEfficiency {

    //===============Tunable parameters============
    private static int batchSize = 50;
    private static int seed = 123;
    private static double trainFraction = 0.8;
    private static double lr = 0.001;
    //=============================================

    public static void main(String[] args) throws IOException, InterruptedException {

        /*
         * Step 1: Read the CSV file in a RecordReader
         */
        //====== Your code block starts here===========

        File dataFile = new ClassPathResource("ENB2012_data.csv").getFile();
        RecordReader rr = new CSVRecordReader(1,',');
        FileSplit filesplit = new FileSplit(dataFile);
        rr.initialize(filesplit);

        //====== Your code block ends here===========

        Schema schema = new Schema.Builder()
                .addColumnsDouble("rel-compactness","surface-area","wall-area","roof-area","overall-height")
                .addColumnInteger("orientation")
                .addColumnDouble("glazing-area")
                .addColumnInteger("glazing-area-distribution")
                .addColumnsDouble("heating-load","cooling-load")
                .addColumnsString("emptyCol1","emptyCol2")
                .build();

        /*
         * Step 2: Specify the transform process
         * 2-1: remove unused columns
         * 2-2: filter out empty rows
         */
        //====== Your code block starts here===========

        TransformProcess tp = new TransformProcess.Builder(schema)
                .removeColumns("emptyCol1","emptyCol2")
                .filter(new FilterInvalidValues())
                .build();

        //====== Your code block ends here===========

        List<List<Writable>> data = new ArrayList<>();

        /*
         * Step 3: Put each instances into the List<List<Writable>>
         * */
        //====== Your code block starts here===========

        while (rr.hasNext()){
            List<Writable> Data = rr.next();
            data.add(Data);
        }

        //====== Your code block ends here===========

        List<List<Writable>> transformed = LocalTransformExecutor.execute(data, tp);
        System.out.println("=======Initial Schema=========\n"+ tp.getInitialSchema());
        System.out.println("=======Final Schema=========\n"+ tp.getFinalSchema());

        /*
         * Check the size of array before and after transformation
         * Please ensure that Pre-transformation size > Post-transformation size
         */
        System.out.println("Size before transform: "+ data.size()+
                "\nColumns before transform: "+ tp.getInitialSchema().numColumns());
        System.out.println("Size after transform: "+ transformed.size()+
                "\nColumns after transform: "+ tp.getFinalSchema().numColumns());

        CollectionRecordReader crr = new CollectionRecordReader(transformed);

        /*
         * Step 4: Perform shuffling of dataset with seed number provided
         * Do keep in mind that there are 2 targets for this regression problem
         */
        //====== Your code block starts here===========

        DataSetIterator iter = new RecordReaderDataSetIterator(crr,transformed.size(),8,9,true);
        DataSet dataset = iter.next();
        dataset.shuffle(seed);

        //====== Your code block ends here===========

        /*
         * Step 5: Do train and test splitting
         */
        //====== Your code block starts here===========

        SplitTestAndTrain TestnTrain = dataset.splitTestAndTrain(trainFraction);
        DataSet trainSet = TestnTrain.getTrain();
        DataSet testSet = TestnTrain.getTest();

        //====== Your code block ends here===========

        /*
         * Step 6: Do data normalisation in the form of [0,1]
         */
        //====== Your code block starts here===========

        DataNormalization normalizer = new NormalizerMinMaxScaler();
        normalizer.fit(trainSet);
        normalizer.transform(trainSet);
        normalizer.transform(testSet);

        //====== Your code block ends here===========

        ViewIterator trainIter = new ViewIterator(trainSet,batchSize);
        ViewIterator testIter = new ViewIterator(testSet, batchSize);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .updater(new Adam(lr))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(trainIter.inputColumns())
                        .nOut(50)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nOut(2)
                        .activation(Activation.SIGMOID)
                        .build())
                .build();

        /*
         * Step 7: Perform early stopping so that the model does not overfit
         * Specifications:-
         * a) Calculate average loss on dataset
         * b) terminate if there is no improvement for 1 epoch
         * c) check the loss for each epoch
         * Your model should not take more than 200 epochs for training
         */
        //====== Your code block starts here===========


        EarlyStoppingConfiguration esConfig = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(200))
                .scoreCalculator(new DataSetLossCalculator(testIter,true))
                .evaluateEveryNEpochs(1)
                .build();

        /*
         * Step 8: Perform evaluation for both train and test set using the early stopping model
         */

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConfig,config,trainIter);

        EarlyStoppingResult result = trainer.fit();

        MultiLayerNetwork model = new MultiLayerNetwork(config);

        //Set model listeners
        model.init();
        UIServer server = UIServer.getInstance();
        StatsStorage storage = new InMemoryStatsStorage();
        server.attach(storage);
        model.setListeners(new ScoreIterationListener(1), new StatsListener(storage));

        //Set the number of epoch using the best results from the first Early Stopping training to retrain the model
        //result.getBestModelEpoch() - the optimal epoch number
        System.out.println("Training model ...");
        System.out.println("Best Epochs is " + result.getBestModelEpoch());

        for(int i = 0; i < result.getBestModelEpoch(); i++) {
            model.fit(trainIter);
            Evaluation eval = model.evaluate(testIter);
            System.out.println("EPOCH: " + i + " Accuracy: " + eval.accuracy());
        }

        Evaluation evalTrain = model.evaluate(trainIter);
        Evaluation evalTest = model.evaluate(trainIter);

        //====== Your code block ends here===========

        System.out.println(evalTrain.stats());
        System.out.println(evalTest.stats());

    }

}