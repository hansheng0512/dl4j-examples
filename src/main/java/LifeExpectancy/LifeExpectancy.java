package LifeExpectancy;

/*
 * Before you start, you should:
 * 1. Put the CSV file into your resources folder

 * Here, the hints of what you should do is given below:

 * Step 1: Read the CSV file in a RecordReader.
 * Step 2: Specify the transform process.
 * Step 3: Put each instances into the List<List<Writable>>.
 * Step 4: Execute the transform process.
 * Step 5: Perform shuffling of dataset with seed number provided.
 * Step 6: Split the dataset using train fraction provided and assign the large fraction as trainAndVal
 *         while the another as test split.
 * Step 7: Define feature scaling. Try to experiment with both feature scaling methods.
 * Step 8: Perform scaling only on trainAndVal
 * Step 9: Perform K-fold cross validation. Take k=5.
 * Step 10: Write the model config in the static method and return it in the main method.
 * Step 11: Perform training
 * Step 12: Perform evaluation on trainAndVal and test set

 */


import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.filter.FilterInvalidValues;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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
import org.datavec.api.transform.schema.Schema;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.xml.crypto.Data;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class LifeExpectancy {

    private static int seed = 12345;
    private static double lr = 0.001;
    private static int epoch = 10;
    private static double trainPerc = 0.7;

    public static void main(String[] args) throws Exception {

        /*
         * Step 1: Read the CSV file in a RecordReader
         */
        //====== Your code block starts here===========

        File dataFile = new ClassPathResource("life_exp.csv").getFile();
        RecordReader rr = new CSVRecordReader(1, ',');
        FileSplit filesplit = new FileSplit(dataFile);
        rr.initialize(filesplit);

        //====== Your code block ends here===========

        //define schema
        Schema schema = new Schema.Builder()
                .addColumnsDouble("country")
                .addColumnInteger("year")
                .addColumnString("status")
                .addColumnDouble("life-exp")
                .addColumnsInteger("adult-mortality","infant-death")
                .addColumnsDouble("alcohol","perc-expenditure")
                .addColumnsInteger("hepa-b","measles")
                .addColumnDouble("bmi")
                .addColumnsInteger("under-5-deaths","polio")
                .addColumnDouble("total-expenditure")
                .addColumnInteger("diphtheria")
                .addColumnsDouble("hiv-aids","gdp")
                .addColumnInteger("population")
                .addColumnsDouble("thinness-1_19","thinness-5_9","income-composition","schooling")
                .build();

        /*
         * Step 2: Specify the transform process
         * 2-1: convert "status" into categorical - "developing" and "developed"
         * 2-2: convert "status" to one-hot
         *
         */
        TransformProcess tp = new TransformProcess.Builder(schema)
                .removeColumns("country")
                //====== Your code block starts here===========
                .stringToCategorical("status", Arrays.asList("Developing", "Developed"))
                .categoricalToOneHot("status")
                //====== Your code block ends here===========
                .filter(new FilterInvalidValues())
                .build();

        List<List<Writable>> original = new ArrayList<>();

        /*
         * Step 3: Put each instances into the List<List<Writable>>
         * */
        //====== Your code block starts here===========

        while(rr.hasNext()){
            original.add(rr.next());
        }

        //====== Your code block ends here===========

        /*
         * Step 4: Execute transform process
         * */
        //====== Your code block starts here===========

        List<List<Writable>> transformed = LocalTransformExecutor.execute(original, tp);

        //====== Your code block ends here===========

        System.out.println("Schema before transformed: \n"+tp.getInitialSchema());
        System.out.println("Schema after transformed: \n"+tp.getFinalSchema());

        System.out.println("Size before transformed: "+original.size());
        System.out.println("Size after transformed: "+transformed.size());
        System.out.println("Columns before transformed: "+tp.getInitialSchema().numColumns());
        System.out.println("Columns after transformed: "+tp.getFinalSchema().numColumns());

        CollectionRecordReader crr = new CollectionRecordReader(transformed);

        /*
         * Step 5: Perform shuffling of dataset with seed number provided
         * Keep in mind that there is only 1 output
         * */
        //====== Your code block starts here===========

        DataSetIterator iter = new RecordReaderDataSetIterator(crr,transformed.size(),3,3,true);
        DataSet dataset = iter.next();
        dataset.shuffle(seed);

        //====== Your code block ends here===========

        /*
         * Step 6: Split the dataset using train fraction provided and
         *         assign the large fraction as trainAndVal while the another as test split
         * Keep in mind that there is only 1 output
         * */
        //====== Your code block starts here===========

        SplitTestAndTrain TrainTest = dataset.splitTestAndTrain(trainPerc);
        DataSet train  =TrainTest.getTrain();
        DataSet test  =TrainTest.getTest();

        //====== Your code block ends here===========

        /*
         * Step 7: Define feature scaling
         * hint: experiment with both types of scaling method
         * */
        //====== Your code block starts here===========

        DataNormalization scaler = new NormalizerMinMaxScaler(0,1);

        //====== Your code block ends here===========

        //perform scaling only on train and validation set
        /*
         * Step 8: Perform scaling only on trainAndVal
         *
         * */
        //====== Your code block starts here===========

        scaler.fit(test);

        //====== Your code block ends here===========

        ViewIterator testIter = new ViewIterator(test,test.numExamples());
        testIter.setPreProcessor(scaler);

        /*
         * Step 9: Perform K-fold cross validation
         * Take K=5
         * */
        //====== Your code block starts here===========

        KFoldIterator kfloat = new KFoldIterator(5,test);

        //====== Your code block ends here===========

        /*
         * Step 10: Write the model config in the static method and return it in the main method
         * Keep in mind to use the following input parameters provided: seed and learning rate
         * */
        //====== Your code block starts here===========

        ViewIterator trainIter = new ViewIterator(train,128);
        MultiLayerConfiguration config = getConfig(seed,lr,trainIter.inputColumns(),2);

        //====== Your code block ends here===========

        MultiLayerNetwork model = new MultiLayerNetwork(config);

        InMemoryStatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);
        model.setListeners(new StatsListener(storage,1),new ScoreIterationListener(10));

        /*
         * Step 11: Perform training
         * */
        model.init();
        //====== Your code block starts here===========

        //====== Your code block ends here===========
        System.out.println(model.summary());

        /*
         * Step 12: Perform evaluation on trainAndVal and test set
         * */
        //====== Your code block starts here===========

        Evaluation evalTrain = model.evaluate(trainIter);
        Evaluation evalTest = model.evaluate(kfloat);

        //====== Your code block ends here===========
        System.out.println("Train & Validation evaluation\n"+evalTrain.stats());
        System.out.println("Test evaluation\n"+evalTest.stats());

    }

    private static MultiLayerConfiguration getConfig(int seedNum, double learningRate, int nFeatures, int nClass){

        /*
          Step 10-1: write the config using the input variables and complete the static method
          Specifications:
          1. use Adam
          2. use Xavier
          3. use 1 hidden layer with 100 neurons
          4. use MSE as loss function
         */
        //====== Your code block starts here===========

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seedNum)
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(nFeatures)
                        .activation(Activation.RELU)
                        .nOut(100)
                        .build()
                )
                .layer(1,new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nOut(1)
                        .build()
                )
                .build();

        return config;

        //====== Your code block ends here===========
    }


}