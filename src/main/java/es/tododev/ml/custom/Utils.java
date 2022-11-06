package es.tododev.ml.custom;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.Random;
import java.util.zip.ZipInputStream;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;

public class Utils {

	public static double sigmoid(double x) {
		return (1 / (1 + Math.pow(Math.E, (-1 * x))));
	}
	
	public static List<Data> getFromCSV(InputStream is) throws FileNotFoundException, IOException{
		List<Data> dataset = new ArrayList<>();
		try(ZipInputStream zipIn = new ZipInputStream(is)){
			zipIn.getNextEntry();
			try(BufferedReader reader = new BufferedReader(new InputStreamReader(zipIn))){
				CSVParser parser = CSVFormat.DEFAULT.parse(reader);
				parser.forEach(record -> {
					double[] values = new double[record.size()-1];
					for(int i=1;i<record.size();i++) {
						values[i-1] = Double.parseDouble(record.get(i));
					}
					Data data = new Data(record.get(0), values);
					dataset.add(data);
				});
			}
			return dataset;

		}
		
	}
	
	public static List<Data> getFromCSV(String zip) throws FileNotFoundException, IOException{
		try(InputStream is = new FileInputStream(zip)){
			return getFromCSV(is);
		}
	}
	
	public static double[] normalize(double[] rawData) {
		DoubleSummaryStatistics stats = Arrays.stream(rawData).summaryStatistics();
		double max = stats.getMax();
		double[] normalized = new double[rawData.length];
		for(int i=1;i<rawData.length;i++) {
			normalized[i] = rawData[i] / max;
		}
		return normalized;
	}
	
	public static double getRandom(double rangeMin, double rangeMax) {
		Random r = new Random();
		return rangeMin + (rangeMax - rangeMin) * r.nextDouble();
	}

}
