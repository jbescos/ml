package es.tododev.ml.util;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

import javax.imageio.ImageIO;

import com.google.common.primitives.Doubles;
import com.google.common.primitives.Floats;

public class ImageToCsv {
    
    private static final int WHITE_RGB = 0xffffffff;

    public static float[] imageToArray(InputStream in) throws IOException {
        boolean blank = true;
        BufferedImage image = ImageIO.read(in);
        int size = image.getHeight() * image.getWidth();
        float[] pixels = new float[size];
        int idx = 0;
        for (int i = 0; i < image.getHeight(); i++) {
            for (int j = 0; j < image.getWidth(); j++) {
                int pixel = image.getRGB(j, i);
                if (WHITE_RGB == pixel) {
                    pixels[idx] = 0.0f;
                } else {
                    pixels[idx] = 1.0f;
                    blank = false;
                }
                idx++;
            }
        }
        if (blank) {
            throw new IllegalStateException("Image is blank");
        }
//        System.out.println("Image: " + Arrays.toString(pixels));
        return pixels;
    }

    
    public static double[] imageToArrayD(InputStream in) throws IOException {
        float[] f = imageToArray(in);
        return Doubles.toArray(Floats.asList(f));
    }
}
