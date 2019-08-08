package es.tododev.ml;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.google.common.collect.Lists;

public class Model implements IModel {
	
	private static final long serialVersionUID = 1L;
	private final List<INeuron[]> layers = new ArrayList<>();
	private static final int ELEMENTS_TO_CALCULATE_COST = 100;
	
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
//		for(int i=0;i<iterations;i++) {
		Collections.shuffle(trainingData);
		for(List<Data> data : Lists.partition(trainingData, ELEMENTS_TO_CALCULATE_COST)) {
			backPropagate(data);
		}
		double cost = obtainCost(trainingData);
		System.out.println("Cost "+cost);
//		}
		
		
	}
	
	private void backPropagate(List<Data> datas) {
		for(Data data : datas) {
			INeuron[] outputLayer = getResult(data.getInputValues());
			// TODO
		}
	}
	
	private void setRandomWeights() {
		for(int i=1;i<layers.size();i++) {
			INeuron[] neurons = layers.get(i);
			for(int j=0;j<neurons.length;j++) {
				INeuron neuron = neurons[j];
				for(int w=0;w<neuron.getInputWeights().length;w++) {
					neuron.getInputWeights()[w] = Utils.getRandom(0, 1);
				}
			}
		}
	}
	
	private double obtainCost(List<Data> datas) {
		int total = datas.size();
		double totalCost = 0;
		for(Data data : datas) {
			INeuron[] outputLayer = getResult(data.getInputValues());
			totalCost = calculateCost(data.getExpectedLabel(), outputLayer) + totalCost;
		}
		totalCost = totalCost / total;
		return totalCost;
	}
	
	double calculateCost(String expectedLabel, INeuron[] outputLayer) {
		double total = 0;
		for(INeuron neuron : outputLayer) {
			double diff = neuron.getValue() - (expectedLabel.equals(neuron.getLabel()) ? 1 : 0);
			total = Math.pow(diff, 2) + total;
		}
		return total / outputLayer.length;
	}

	private void addOutput(String ... labels) {
		INeuron[] layer = new INeuron[labels.length];
		for(int i=0;i<labels.length;i++) {
			layer[i] = new Neuron(labels[i]);
		}
		addLayer(layer);
	}

	private void addLayer(int neurons) {
		INeuron[] layer = new INeuron[neurons];
		for(int i=0;i<neurons;i++) {
			layer[i] = new Neuron();
		}
		addLayer(layer);
	}
	
	@Override
	public void setLayers(int[] layersSize, String[] outputLayer) {
		for(int size : layersSize) {
			addLayer(size);
		}
		addOutput(outputLayer);
		setRandomWeights();
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

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		for(int i=1;i<layers.size();i++) {
			builder.append("Layer ").append(i).append("\n");
			INeuron[] neurons = layers.get(i);
			for(int j=0;j<neurons.length;j++) {
				INeuron neuron = neurons[j];
				builder.append("Neuron ").append(j).append(" ").append(neuron.getLabel()).append(" = ").append(neuron.getValue());
				builder.append(" [ ");
				for(int w=0;w<neuron.getInputWeights().length;w++) {
					builder.append(neuron.getInputWeights()[w]).append(" ");
				}
				builder.append("]");
				builder.append("\n");
			}
		}
		
		return builder.toString();
	}
	
	
}
