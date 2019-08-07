package es.tododev.ml;

import java.io.Serializable;

public interface INeuron extends Serializable {

	void execute();
	double[] getInputWeights();
	INeuron[] getInputNeurons();
	double getValue();
	void setValue(double value);
	void setBias(double bias);
	double getBias();
	void setInputWeights(double[] inputWeights);
	void setInputNeurons(INeuron[] inputNeurons);
	void setLabel(String label);
	String getLabel();
	void trainPaths(double weightToIncrement);
	
}
