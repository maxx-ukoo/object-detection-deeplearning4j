package ramo.klevis;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.deeplearning4j.zoo.model.YOLO2;
import org.deeplearning4j.zoo.model.YOLO2.YOLO2Builder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import com.google.protobuf.TextFormat;

import maxx.test.protos.StringIntLabelMapOuterClass.StringIntLabelMap;
import maxx.test.protos.StringIntLabelMapOuterClass.StringIntLabelMapItem;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Optional;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_highgui.imshow;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;

public class TinyYoloPrediction {

	private ComputationGraph preTrained;
	private List<DetectedObject> predictedObjects;
	private HashMap<Integer, String> map;
	private static double detectionThreshold = 0.4;

	private TinyYoloPrediction() {

		try {
			preTrained = (ComputationGraph) YOLO2.builder().build().initPretrained();
			prepareLabels();
			preTrained.summary();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		//preTrained = TinyYOLO.builder().build().init();
		// preTrained = (ComputationGraph) new TinyYOLO().initPretrained();


	}

	private static final TinyYoloPrediction INSTANCE = new TinyYoloPrediction();

	public static TinyYoloPrediction getINSTANCE() {
		return INSTANCE;
	}

	public void markWithBoundingBox(Mat file, int imageWidth, int imageHeight, boolean newBoundingBOx, String winName)
			throws Exception {
		int width = 416;
		int height = 416;
		int gridWidth = 13;
		int gridHeight = 13;

		Yolo2OutputLayer outputLayer = (Yolo2OutputLayer) preTrained.getOutputLayer(0);
		if (newBoundingBOx) {
			INDArray indArray = prepareImage(file, width, height);
			INDArray results = preTrained.outputSingle(indArray);
			predictedObjects = outputLayer.getPredictedObjects(results, detectionThreshold);
			// System.out.println("results = " + predictedObjects);
			markWithBoundingBox(file, gridWidth, gridHeight, imageWidth, imageHeight);
		} else {
			markWithBoundingBox(file, gridWidth, gridHeight, imageWidth, imageHeight);
		}
		imshow(winName, file);
	}

	private INDArray prepareImage(Mat file, int width, int height) throws IOException {
		NativeImageLoader loader = new NativeImageLoader(height, width, 3);
		ImagePreProcessingScaler imagePreProcessingScaler = new ImagePreProcessingScaler(0, 1);
		INDArray indArray = loader.asMatrix(file);
		imagePreProcessingScaler.transform(indArray);
		return indArray;
	}

	private void prepareLabels() {
		if (map == null) {
			String s = "person\n" + 
					"bicycle\n" + 
					"car\n" + 
					"motorbike\n" + 
					"aeroplane\n" + 
					"bus\n" + 
					"train\n" + 
					"truck\n" + 
					"boat\n" + 
					"traffic light\n" + 
					"fire hydrant\n" + 
					"stop sign\n" + 
					"parking meter\n" + 
					"bench\n" + 
					"bird\n" + 
					"cat\n" + 
					"dog\n" + 
					"horse\n" + 
					"sheep\n" + 
					"cow\n" + 
					"elephant\n" + 
					"bear\n" + 
					"zebra\n" + 
					"giraffe\n" + 
					"backpack\n" + 
					"umbrella\n" + 
					"handbag\n" + 
					"tie\n" + 
					"suitcase\n" + 
					"frisbee\n" + 
					"skis\n" + 
					"snowboard\n" + 
					"sports ball\n" + 
					"kite\n" + 
					"baseball bat\n" + 
					"baseball glove\n" + 
					"skateboard\n" + 
					"surfboard\n" + 
					"tennis racket\n" + 
					"bottle\n" + 
					"wine glass\n" + 
					"cup\n" + 
					"fork\n" + 
					"knife\n" + 
					"spoon\n" + 
					"bowl\n" + 
					"banana\n" + 
					"apple\n" + 
					"sandwich\n" + 
					"orange\n" + 
					"broccoli\n" + 
					"carrot\n" + 
					"hot dog\n" + 
					"pizza\n" + 
					"donut\n" + 
					"cake\n" + 
					"chair\n" + 
					"sofa\n" + 
					"pottedplant\n" + 
					"bed\n" + 
					"diningtable\n" + 
					"toilet\n" + 
					"tvmonitor\n" + 
					"laptop\n" + 
					"mouse\n" + 
					"remote\n" + 
					"keyboard\n" + 
					"cell phone\n" + 
					"microwave\n" + 
					"oven\n" + 
					"toaster\n" + 
					"sink\n" + 
					"refrigerator\n" + 
					"book\n" + 
					"clock\n" + 
					"vase\n" + 
					"scissors\n" + 
					"teddy bear\n" + 
					"hair drier\n" + 
					"toothbrush";
			/*String s = "aeroplane\n" + "bicycle\n" + "bird\n" + "boat\n" + "bottle\n" + "bus\n" + "car\n" + "cat\n"
					+ "chair\n" + "cow\n" + "diningtable\n" + "dog\n" + "horse\n" + "motorbike\n" + "person\n"
					+ "pottedplant\n" + "sheep\n" + "sofa\n" + "train\n" + "tvmonitor";*/
			String[] split = s.split("\\n");
			int i = 0;
			map = new HashMap<>();
			for (String s1 : split) {
				map.put(i++, s1);
			}
		}
	}



	private void markWithBoundingBox(Mat file, int gridWidth, int gridHeight, int w, int h) {

		if (predictedObjects == null) {
			return;
		}
		ArrayList<DetectedObject> detectedObjects = new ArrayList<>(predictedObjects);

		while (!detectedObjects.isEmpty()) {
			Optional<DetectedObject> max = detectedObjects.stream()
					.max((o1, o2) -> ((Double) o1.getConfidence()).compareTo(o2.getConfidence()));
			if (max.isPresent()) {
				DetectedObject maxObjectDetect = max.get();
				removeObjectsIntersectingWithMax(detectedObjects, maxObjectDetect);
				detectedObjects.remove(maxObjectDetect);
				markWithBoundingBox(file, gridWidth, gridHeight, w, h, maxObjectDetect);
			}
		}
	}

	private static void removeObjectsIntersectingWithMax(ArrayList<DetectedObject> detectedObjects,
			DetectedObject maxObjectDetect) {
		double[] bottomRightXY1 = maxObjectDetect.getBottomRightXY();
		double[] topLeftXY1 = maxObjectDetect.getTopLeftXY();
		List<DetectedObject> removeIntersectingObjects = new ArrayList<>();
		for (DetectedObject detectedObject : detectedObjects) {
			double[] topLeftXY = detectedObject.getTopLeftXY();
			double[] bottomRightXY = detectedObject.getBottomRightXY();
			double iox1 = Math.max(topLeftXY[0], topLeftXY1[0]);
			double ioy1 = Math.max(topLeftXY[1], topLeftXY1[1]);

			double iox2 = Math.min(bottomRightXY[0], bottomRightXY1[0]);
			double ioy2 = Math.min(bottomRightXY[1], bottomRightXY1[1]);

			double inter_area = (ioy2 - ioy1) * (iox2 - iox1);

			double box1_area = (bottomRightXY1[1] - topLeftXY1[1]) * (bottomRightXY1[0] - topLeftXY1[0]);
			double box2_area = (bottomRightXY[1] - topLeftXY[1]) * (bottomRightXY[0] - topLeftXY[0]);

			double union_area = box1_area + box2_area - inter_area;
			double iou = inter_area / union_area;

			if (iou > detectionThreshold) {
				removeIntersectingObjects.add(detectedObject);
			}

		}
		detectedObjects.removeAll(removeIntersectingObjects);
	}

	private void markWithBoundingBox(Mat file, int gridWidth, int gridHeight, int w, int h, DetectedObject obj) {

		double[] xy1 = obj.getTopLeftXY();
		double[] xy2 = obj.getBottomRightXY();
		int predictedClass = obj.getPredictedClass();
		int x1 = (int) Math.round(w * xy1[0] / gridWidth);
		int y1 = (int) Math.round(h * xy1[1] / gridHeight);
		int x2 = (int) Math.round(w * xy2[0] / gridWidth);
		int y2 = (int) Math.round(h * xy2[1] / gridHeight);
		rectangle(file, new Point(x1, y1), new Point(x2, y2), Scalar.RED);
		putText(file, map.get(predictedClass) + String.format("%f", obj.getConfidence()), new Point(x1 + 2, y2 - 2),
				FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN);
	}

}