package es.tododev.ml;

public class Neuron implements INeuron {

	private static final long serialVersionUID = 1L;
	private double value;
	private double bias;
	private double[] inputWeights;
	private INeuron[] inputNeurons;
	private String label;
	
	public Neuron() {
	}
	public Neuron(String label) {
		this.label = label;
	}
	@Override
	public double getValue() {
		return value;
	}
	@Override
	public void setValue(double value) {
		this.value = value;
	}
	@Override
	public double getBias() {
		return bias;
	}
	@Override
	public void setBias(double bias) {
		this.bias = bias;
	}
	@Override
	public double[] getInputWeights() {
		return inputWeights;
	}
	@Override
	public void setInputWeights(double[] inputWeights) {
		this.inputWeights = inputWeights;
	}
	@Override
	public INeuron[] getInputNeurons() {
		return inputNeurons;
	}
	@Override
	public void setInputNeurons(INeuron[] inputNeurons) {
		this.inputNeurons = inputNeurons;
	}
	@Override
	public void setLabel(String label) {
		this.label = label;
	}
	@Override
	public String getLabel() {
		return label;
	}
	@Override
	public void execute() {
		for(int i=0;i<inputNeurons.length;i++) {
			value = value + (inputNeurons[i].getValue() * inputWeights[i]);
		}
		value = Utils.sigmoid((value/inputNeurons.length) + bias);
	}
	@Override
	public String toString() {
		return "["+label+" = "+value+"]";
	}
	
}
