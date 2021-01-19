//package EnergyEfficiency;
//
//import org.datavec.api.records.reader.RecordReader;
//import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
//import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
//import org.datavec.api.split.FileSplit;
//import org.datavec.api.transform.TransformProcess;
//import org.datavec.api.transform.schema.Schema;
//import org.datavec.api.writable.Writable;
//import org.datavec.local.transforms.LocalTransformExecutor;
//import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
//import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
//import org.deeplearning4j.nn.conf.layers.DenseLayer;
//import org.deeplearning4j.nn.conf.layers.OutputLayer;
//import org.deeplearning4j.nn.weights.WeightInit;
//import org.nd4j.common.io.ClassPathResource;
//import org.nd4j.linalg.activations.Activation;
//import org.nd4j.linalg.dataset.ViewIterator;
//
//import java.io.File;
//import java.io.IOException;
//import java.util.ArrayList;
//import java.util.List;
//
//public class train {
//
//    //===============Tunable parameters============
//    private static int batchSize = 50;
//    private static int seed = 123;
//    private static double trainFraction = 0.8;
//    private static double lr = 0.001;
//    //=============================================
//
//    public static void main(String[] args) throws Exception {
//        /*
//         * Step 1: Read the CSV file in a RecordReader
//         */
//        //====== Your code block starts here===========
//
//        File dataFile = new ClassPathResource("ENB2012_data.csv").getFile();
//        FileSplit fileSplit = new FileSplit(dataFile);
//
//        RecordReader rr = new CSVRecordReader(1,',');
//        rr.initialize(fileSplit);
//
//        //====== Your code block ends here===========
//
//        Schema schema = new Schema.Builder()
//                .addColumnsDouble("rel-compactness","surface-area","wall-area","roof-area","overall-height")
//                .addColumnInteger("orientation")
//                .addColumnDouble("glazing-area")
//                .addColumnInteger("glazing-area-distribution")
//                .addColumnsDouble("heating-load","cooling-load")
//                .addColumnsString("emptyCol1","emptyCol2")
//                .build();
//
//        /*
//         * Step 2: Specify the transform process
//         * 2-1: remove unused columns
//         * 2-2: filter out empty rows
//         */
//        //====== Your code block starts here===========
//
//        TransformProcess tp = new TransformProcess.Builder(schema)
//                .removeColumns()
//
//
//        //====== Your code block ends here===========
//
//        List<List<Writable>> data = new ArrayList<>();
//
//        /*
//         * Step 3: Put each instances into the List<List<Writable>>
//         * */
//        //====== Your code block starts here===========
//
//
//
//        //====== Your code block ends here===========
//
//        List<List<Writable>> transformed = LocalTransformExecutor.execute(data, tp);
//        System.out.println("=======Initial Schema=========\n"+ tp.getInitialSchema());
//        System.out.println("=======Final Schema=========\n"+ tp.getFinalSchema());
//
//        /*
//         * Check the size of array before and after transformation
//         * Please ensure that Pre-transformation size > Post-transformation size
//         */
//        System.out.println("Size before transform: "+ data.size()+
//                "\nColumns before transform: "+ tp.getInitialSchema().numColumns());
//        System.out.println("Size after transform: "+ transformed.size()+
//                "\nColumns after transform: "+ tp.getFinalSchema().numColumns());
//
//        CollectionRecordReader crr = new CollectionRecordReader(transformed);
//
//        /*
//         * Step 4: Perform shuffling of dataset with seed number provided
//         * Do keep in mind that there are 2 targets for this regression problem
//         */
//        //====== Your code block starts here===========
//
//        //====== Your code block ends here===========
//
//        /*
//         * Step 5: Do train and test splitting
//         */
//        //====== Your code block starts here===========
//
//        //====== Your code block ends here===========
//
//        /*
//         * Step 6: Do data normalisation in the form of [0,1]
//         */
//        //====== Your code block starts here===========
//
//        //====== Your code block ends here===========
//
//        ViewIterator trainIter = new ViewIterator(trainSet, batchSize);
//        ViewIterator testIter = new ViewIterator(testSet, batchSize);
//
//        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
//                .seed(seed)
//                .weightInit(WeightInit.XAVIER)
//                .activation(Activation.RELU)
//                .updater(new Adam(lr))
//                .list()
//                .layer(new DenseLayer.Builder()
//                        .nIn(trainIter.inputColumns())
//                        .nOut(50)
//                        .build())
//                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
//                        .nOut(2)
//                        .activation(Activation.IDENTITY)
//                        .build())
//                .build();
//
//        /*
//         * Step 7: Perform early stopping so that the model does not overfit
//         * Specifications:-
//         * a) Calculate average loss on dataset
//         * b) terminate if there is no improvement for 1 epoch
//         * c) check the loss for each epoch
//         * Your model should not take more than 200 epochs for training
//         */
//        //====== Your code block starts here===========
//
//        //====== Your code block ends here===========
//
//        /*
//         * Step 8: Perform evaluation for both train and test set using the early stopping model
//         */
//        //====== Your code block starts here===========
//
//        //====== Your code block ends here===========
//
//        System.out.println(evalTrain.stats());
//        System.out.println(evalTest.stats());
//
//    }
//
//}