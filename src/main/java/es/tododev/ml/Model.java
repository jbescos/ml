package es.tododev.ml;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Model implements IModel {
	
	private static final long serialVersionUID = 1L;
	private final List<INeuron[]> layers = new ArrayList<>();
	
	private void addLayer(INeuron[] neurons) {
		if(!layers.isEmpty()) {
			INeuron[] previousLayer = layers.get(layers.size() - 1);
			for(INeuron currentNeuron : neurons) {
				currentNeuron.setInputNeurons(previousLayer);
				double[] inputWeights = new double[previousLayer.length];
				currentNeuron.setInputWeights(inputWeights);
			}
		}
		layers.add(neurons);
	}

	@Override
	public INeuron[] getResult(double[] inputValues) {
		double[] normalizedInput = Utils.normalize(inputValues);
		INeuron[] inputLayer = layers.get(0);
		for(int i=0;i<normalizedInput.length;i++) {
			inputLayer[i].setValue(normalizedInput[i]);
		}
		for(int i=1;i<layers.size();i++) {
			for(INeuron neuron : layers.get(i)) {
				neuron.execute();
			}
		}
		return layers.get(layers.size() - 1);
	}

	@Override
	public void train(int iterations, List<Data> trainingData) {
		for(int i=0;i<iterations;i++) {
			for(Data data : trainingData) {
				computeData(data);
			}
		}
		
		
	}
	
	private void computeData(Data data) {
		INeuron[] inputLayer = layers.get(0);
		INeuron[] outputLayer = layers.get(layers.size()-1);
		double[] normalizedInput = Utils.normalize(data.getInputValues());
		for(int i=0;i<inputLayer.length;i++) {
			inputLayer[i].setValue(normalizedInput[i]);
		}
		for(INeuron outNeuron : outputLayer) {
			if(data.getExpectedLabel().equals(outNeuron.getLabel())) {
				outNeuron.trainPaths(1);
			}else {
				outNeuron.trainPaths(-1);
			}
		}
	}

	@Override
	public void addOutput(String ... labels) {
		INeuron[] layer = new INeuron[labels.length];
		for(int i=0;i<labels.length;i++) {
			layer[i] = new Neuron(labels[i]);
		}
		addLayer(layer);
	}

	@Override
	public void addLayer(int neurons) {
		INeuron[] layer = new INeuron[neurons];
		for(int i=0;i<neurons;i++) {
			layer[i] = new Neuron();
		}
		addLayer(layer);
	}

	@Override
	public String getResultLabel(double[] inputValues) {
		double result = 0;
		INeuron winner = null;
		INeuron[] outputLayer = getResult(inputValues);
		for(INeuron neuron : outputLayer) {
			if(neuron.getValue() >= result) {
				result = neuron.getValue();
				winner = neuron;
			}
		}
		return winner.getLabel();
	}
	
}
