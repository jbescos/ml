package es.tododev.ml.custom;

import java.io.Serializable;
import java.util.List;

public interface IModel extends Serializable {

	INeuron[] getResult(double[] inputValues);
	String getResultLabel(double[] inputValues);
	void train(int iterations, List<Data> tests);
	void setLayers(int[] layersSize, String[] outputLayer);
	
}
