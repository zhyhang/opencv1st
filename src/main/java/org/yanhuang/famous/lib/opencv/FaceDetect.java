package org.yanhuang.famous.lib.opencv;

import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

/**
 * reference example: https://www.baeldung.com/java-opencv
 */
public class FaceDetect {

	private static final String defaultClassierPath = "/code/person/opencv1st/src/main/resources/haarcascades" +
			"/haarcascade_frontalface_alt.xml";
	private static final String exampleSourceImagePath = "/code/person/opencv1st/src/main/resources/images" +
			"/red600x300.jpg";

	private static final String targetImageParentPath = "/temp/";

	static {
		OpenCV.loadShared();
	}

	public static void main(String[] args) throws Exception {
		new FaceDetect().detect(null, null, null);
	}

	public void detect(String sourceImagePath, String classierPath, String targetImagePath) {
		String usedSourceImagePath = Objects.isNull(sourceImagePath) || sourceImagePath.trim().isEmpty() ?
				exampleSourceImagePath : sourceImagePath;
		Mat loadedImage = loadImage(usedSourceImagePath);
		String usedClassierPath = Objects.isNull(classierPath) || classierPath.trim().isEmpty() ? defaultClassierPath
				: classierPath;
		MatOfRect facesDetected = new MatOfRect();
		CascadeClassifier cascadeClassifier = new CascadeClassifier();
		int minFaceSize = Math.round(loadedImage.rows() * 0.1f);
		cascadeClassifier.load(usedClassierPath);
		cascadeClassifier.detectMultiScale(loadedImage,
				facesDetected,
				1.1,
				3,
				Objdetect.CASCADE_SCALE_IMAGE,
				new Size(minFaceSize, minFaceSize),
				new Size()
		);
		Rect[] facesArray = facesDetected.toArray();
		for(Rect face : facesArray) {
			Imgproc.rectangle(loadedImage, face.tl(), face.br(), new Scalar(0, 0, 255), 3);
		}
		saveImage(loadedImage, buildTargetImagePath(usedSourceImagePath, targetImagePath));
	}

	private String buildTargetImagePath(String usedSourceImagePath, String targetImagePath) {

		if (Objects.isNull(targetImagePath) || targetImagePath.trim().isEmpty()) {
			final Path fileName = Paths.get(usedSourceImagePath).getFileName();
			return targetImageParentPath + "target_" + fileName;
		}
		return targetImagePath;
	}

	private Mat loadImage(String path) {
		return Imgcodecs.imread(path);
	}

	public void saveImage(Mat imageMatrix, String targetPath) {
		Imgcodecs.imwrite(targetPath, imageMatrix);
	}
}
