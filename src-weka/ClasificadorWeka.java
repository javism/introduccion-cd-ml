import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

public class ClasificadorWeka {

	public static void main(String[] args) throws Exception {
		/* Ruta al fichero de weka con la base de datos completa. Tendras que cambiarla para indicar
		 * el fichero que quieres cargar. 
		 */
		
		String dataset = "lymph.arff";
		
		// Objetos para la lectura de instancias de la base de datos
		BufferedReader breader = null;
		breader = new BufferedReader(new FileReader(dataset));
		
		Instances train = new Instances(breader);
		train.setClassIndex(train.numAttributes() - 1);
		breader.close();
		
		weka.classifiers.AbstractClassifier myClassifier = null;
		
		/* ------------------------ CAMBIA SOLO ESTO ------------------------------- 
		 * Clasificador: 
		 *  - Puedes elegir aqui cualquiera de los clasificadores de Weka. Solo debes 
		 *    hacer el import adecuado para utilizarlo en weka.classifiers o poner la 
		 *    ruta completa como se muestra en los ejemplos. 
		 *  - Descomenta una de las opciones para probar distintos clasificadores 
		 *  - Consulta la documentacion de Weka para ver qué atributos tienen los 
		 *    distintos modelos y como cambiarlos, por ejemplo:
		 *    http://weka.sourceforge.net/doc.dev/weka/classifiers/functions/MultilayerPerceptron.html
		 *  
		 *  */
		
		 
		//myClassifier = new NaiveBayes();
		
		myClassifier = new weka.classifiers.trees.J48();
		
		//myClassifier = new weka.classifiers.functions.Logistic();

		//myClassifier = new weka.classifiers.functions.MultilayerPerceptron();
		//myClassifier.setHiddenLayers("50");
		/* ------------------------ CAMBIA SOLO HASTA AQUI -------------------------------*/
		
		// Entrenar el clasificador
		myClassifier.buildClassifier(train);

		// Objetos para la evaluacion
		Evaluation eval = new Evaluation(train);
		eval.crossValidateModel(myClassifier, train, 10, new Random(1));
		System.out.println(eval.toSummaryString("\nResultados\n=========", true));
		// También puedes imprimir métricas de evaluación individualmente
		System.out.println(eval.fMeasure(1) + " " + eval.precision(1) + " " + eval.recall(1));

	}

}
