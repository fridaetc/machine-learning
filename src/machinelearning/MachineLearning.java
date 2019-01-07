package machinelearning;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.Classifier;
 
import java.util.HashMap;
import java.util.Map;
import java.util.Iterator;
import java.util.ArrayList;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

import javafx.application.Application;
import javafx.stage.Stage;
import javafx.scene.Scene;
import javafx.scene.Group;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;

/**
 *
 * @author Frida
 */
public class MachineLearning extends Application {

    @Override
    public void start(Stage primaryStage) throws Exception {
        String filePath = new File("").getAbsolutePath();
        
        try {
            BufferedReader reader = new BufferedReader(new FileReader(filePath + "/src/machinelearning/spiral.arff"));

            try {           
                ArffReader arff = new ArffReader(reader, 1000);
                Instances data = arff.getStructure();

                data.setClassIndex(data.numAttributes() - 1);
                
                Instance inst;
                while ((inst = arff.readInstance(data)) != null) {
                  data.add(inst);
                }
                
                Logistic lg = new Logistic();
                MultilayerPerceptron mlp = new MultilayerPerceptron();

                mlp.setLearningRate(0.2); //L
                mlp.setTrainingTime(1000); //N
                mlp.setHiddenLayers("72"); //H
                mlp.setSeed(1); //S
                //mlp.setNormalizeNumericClass(true); //C
                //mlp.setValidationThreshold(0); //E
                //mlp.setMomentum(0.2); //M
                //mlp.setNominalToBinaryFilter(true); //B
                
                HashMap<String, Classifier> classifiers = new HashMap<String, Classifier>();
                classifiers.put("MultilayerPerceptron", mlp);
                classifiers.put("Logistic", lg);
                
                Iterator iterator = classifiers.entrySet().iterator();
                while(iterator.hasNext()) {
                   Map.Entry mentry = (Map.Entry)iterator.next();
                   Classifier c = (Classifier)mentry.getValue();
                   Instances dataIncorrect = arff.getStructure();
                   Instances dataCorrect = arff.getStructure();
                   Instance inst2;
                    
                   while ((inst2 = arff.readInstance(data)) != null) {
                        dataIncorrect.add(inst2);
                        dataCorrect.add(inst2);
                   }
                   
                   try {
                        c.buildClassifier(data);
                        Evaluation eval = new Evaluation(data);
                        eval.evaluateModel(c, data);
                       
                        System.out.print("+++++ " + mentry.getKey() + " +++++");
                        System.out.println(eval.toSummaryString());
                        System.out.println(eval.toMatrixString());
                        System.out.println(eval.toClassDetailsString());
                        
                        //System.out.println("=== Incorrect predictions ===");
                        ArrayList<Prediction> predictions = eval.predictions();
                        
                        for (int i = 0, dataSize = data.size(); i < dataSize; i++) {
                            Instance instance = data.get(i);
                            Prediction prediction = predictions.get(i);

                            if (prediction.actual() != prediction.predicted()) {
                                dataIncorrect.add(instance);
                            } else {
                                dataCorrect.add(instance);
                            }
                        }                        

                        openChart(dataCorrect, dataIncorrect, new Stage(), "Incorrectly Classified: " + mentry.getKey());
                        
                    } catch(Exception e) {
                        System.out.println(e);
                    }
                }
                       
                openChart(data, null, primaryStage, "Scatterplot: All");
            } catch(IOException e) {
                System.out.println(e);
            }
        } catch(FileNotFoundException e) {
            System.out.println(e);            
        }
    }
    
    public static void openChart(Instances data, Instances dataIncorrect, Stage stage, String title) {
        stage.setTitle(title);

        NumberAxis xAxis = new NumberAxis(-1, 1, 0.25); 
        xAxis.setLabel("X");          
        NumberAxis yAxis = new NumberAxis(-1, 1, 0.25); 
        yAxis.setLabel("Y");

        ScatterChart<String, Number> scatterChart = new ScatterChart(xAxis, yAxis); 
        HashMap<String, XYChart.Series> series = new HashMap<String, XYChart.Series>();
        
        int data2Size = 0;
        if(dataIncorrect != null) {
            data2Size = dataIncorrect.numInstances();
        }

        for(int i = 0; i < data.numInstances() + data2Size; i++) {
            String className; 
            Instance row;
            
            if(i > data.numInstances() - 1) {
                row = dataIncorrect.get(i - data.numInstances());
                className = row.stringValue(2) + " incorrect";
            } else {
                row = data.get(i);
                className = row.stringValue(2);
            }

            XYChart.Series serie = series.get(className);

            if (serie == null) {
               serie = new XYChart.Series(); 
               serie.setName(className);
               series.put(className, serie);
            }

            XYChart.Data dot = new XYChart.Data(row.value(0), row.value(1));
            serie.getData().add(dot);
        }

        Iterator iterator = series.entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry pair = (Map.Entry)iterator.next();
            scatterChart.getData().addAll((XYChart.Series)pair.getValue());
        }

        Group group = new Group(scatterChart);
        Scene scene = new Scene(group, 600, 600);
        stage.setScene(scene); 
        stage.show();
    }
    
    public static void main(String[] args) {
        launch(args);
    }
}
