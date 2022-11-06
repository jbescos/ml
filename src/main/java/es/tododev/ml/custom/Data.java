package es.tododev.ml.custom;

public class Data {

	private final String expectedLabel;
	private final double[] inputValues;
	
	public Data(String expectedLabel, double[] inputValues) {
		this.expectedLabel = expectedLabel;
		this.inputValues = inputValues;
	}
	
	public String getExpectedLabel() {
		return expectedLabel;
	}
	public double[] getInputValues() {
		return inputValues;
	}
	
}
