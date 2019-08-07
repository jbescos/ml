package es.tododev.ml;

import java.io.Serializable;
import java.util.List;

public interface IModel extends Serializable {

	void addOutput(String ... labels);
	void addLayer(int neurons);
	INeuron[] getResult(double[] inputValues);
	String getResultLabel(double[] inputValues);
	void train(int iterations, List<Data> tests);
	
}
