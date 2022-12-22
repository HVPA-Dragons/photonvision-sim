/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.vision.pipeline;

import edu.wpi.first.math.VecBuilder;
import edu.wpi.first.math.geometry.Rotation3d;
import edu.wpi.first.math.geometry.Transform3d;
import edu.wpi.first.math.geometry.Translation3d;
import edu.wpi.first.math.util.Units;
import edu.wpi.first.util.WPIUtilJNI;

import java.util.ArrayList;
import java.util.List;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.math.MathUtils;
import org.photonvision.raspi.PicamJNI;
import org.photonvision.vision.apriltag.AprilTagDetectorParams;
import org.photonvision.vision.apriltag.DetectionResult;
import org.photonvision.vision.camera.CameraQuirk;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.pipe.CVPipe.CVPipeResult;
import org.photonvision.vision.pipe.impl.*;
import org.photonvision.vision.pipeline.result.CVPipelineResult;
import org.photonvision.vision.target.TrackedTarget;
import org.photonvision.vision.target.TrackedTarget.TargetCalculationParameters;

@SuppressWarnings("DuplicatedCode")
public class AprilTagPipeline extends CVPipeline<CVPipelineResult, AprilTagPipelineSettings> {
    private static final Logger logger = new Logger(AprilTagPipeline.class, LogGroup.Data);
    
    private final RotateImagePipe rotateImagePipe = new RotateImagePipe();
    private final GrayscalePipe grayscalePipe = new GrayscalePipe();
    private final AprilTagDetectionPipe aprilTagDetectionPipe = new AprilTagDetectionPipe();
    private final CalculateFPSPipe calculateFPSPipe = new CalculateFPSPipe();

    public AprilTagPipeline() {
        settings = new AprilTagPipelineSettings();
    }

    public AprilTagPipeline(AprilTagPipelineSettings settings) {
        this.settings = settings;
    }

    @Override
    protected void setPipeParamsImpl() {
        // Sanitize thread count - not supported to have fewer than 1 threads
        settings.threads = Math.max(1, settings.threads);

        RotateImagePipe.RotateImageParams rotateImageParams =
                new RotateImagePipe.RotateImageParams(settings.inputImageRotationMode);
        rotateImagePipe.setParams(rotateImageParams);

        if (cameraQuirks.hasQuirk(CameraQuirk.PiCam) && PicamJNI.isSupported()) {
            // TODO: Picam grayscale
            PicamJNI.setRotation(settings.inputImageRotationMode.value);
            PicamJNI.setShouldCopyColor(true); // need the color image to grayscale
        }

        AprilTagDetectorParams aprilTagDetectionParams =
                new AprilTagDetectorParams(
                        settings.tagFamily,
                        settings.decimate,
                        settings.blur,
                        settings.threads,
                        settings.debug,
                        settings.refineEdges);

        // TODO (HACK): tag width is Fun because it really belongs in the "target model"
        // We need the tag width for the JNI to figure out target pose, but we need a
        // target model for the draw 3d targets pipeline to work...

        // for now, hard code tag width based on enum value
        double tagWidth;

        // This needs
        switch (settings.targetModel) {
            case k200mmAprilTag:
                {
                    tagWidth = Units.inchesToMeters(3.25 * 2);
                    break;
                }
            case k6in_16h5:
                {
                    tagWidth = Units.inchesToMeters(3 * 2);
                    break;
                }
            default:
                {
                    // guess at 200mm?? If it's zero everything breaks, but it should _never_ be zero. Unless
                    // users select the wrong model...
                    tagWidth = 0.16;
                    break;
                }
        }

        aprilTagDetectionPipe.setParams(
                new AprilTagDetectionPipeParams(
                        aprilTagDetectionParams,
                        frameStaticProperties.cameraCalibration,
                        settings.numIterations,
                        tagWidth));
    }

    @Override
    protected CVPipelineResult process(Frame frame, AprilTagPipelineSettings settings) {
        long sumPipeNanosElapsed = 0L;

        CVPipeResult<Mat> grayscalePipeResult;
        Mat rawInputMat;
        boolean inputSingleChannel = frame.image.getMat().channels() == 1;

        if (inputSingleChannel) {
            rawInputMat = new Mat(PicamJNI.grabFrame(true));
            frame.image.getMat().release(); // release the 8bit frame ASAP.
        } else {
            rawInputMat = frame.image.getMat();
            var rotateImageResult = rotateImagePipe.run(rawInputMat);
            sumPipeNanosElapsed += rotateImageResult.nanosElapsed;
        }

        var inputFrame = new Frame(new CVMat(rawInputMat), frameStaticProperties);

        grayscalePipeResult = grayscalePipe.run(rawInputMat);
        sumPipeNanosElapsed += grayscalePipeResult.nanosElapsed;

        var outputFrame = new Frame(new CVMat(grayscalePipeResult.output), frameStaticProperties);

        List<TrackedTarget> targetList;
        CVPipeResult<List<DetectionResult>> tagDetectionPipeResult;

        // Use the solvePNP Enabled flag to enable native pose estimation
        aprilTagDetectionPipe.setNativePoseEstimationEnabled(settings.solvePNPEnabled);

        tagDetectionPipeResult = aprilTagDetectionPipe.run(grayscalePipeResult.output);
        sumPipeNanosElapsed += tagDetectionPipeResult.nanosElapsed;

        targetList = new ArrayList<>();
        for (DetectionResult detection : tagDetectionPipeResult.output) {
            // TODO this should be in a pipe, not in the top level here (Matt)
            if (detection.getDecisionMargin() < settings.decisionMargin) continue;
            if (detection.getHamming() > settings.hammingDist) continue;

            // populate the target list
            // Challenge here is that TrackedTarget functions with OpenCV Contour
            TrackedTarget target =
                    new TrackedTarget(
                            detection,
                            new TargetCalculationParameters(
                                    false, null, null, null, null, frameStaticProperties));

            double timeStart = WPIUtilJNI.now();
            var targetCorners = target.getTargetCorners().toArray(new Point[0]);
            var ippeTargetCorners = new Point[4];
            ippeTargetCorners[0] = targetCorners[2];
            ippeTargetCorners[1] = targetCorners[3];
            ippeTargetCorners[2] = targetCorners[0];
            ippeTargetCorners[3] = targetCorners[1];
            var imagePoints = new MatOfPoint2f(ippeTargetCorners);
            var rvecs = new ArrayList<Mat>();
            var tvecs = new ArrayList<Mat>();
            Calib3d.solvePnPGeneric(
                new MatOfPoint3f(
                    new Point3(-0.0762, 0.0762, 0),
                    new Point3(0.0762, 0.0762, 0),
                    new Point3(0.0762, -0.0762, 0),
                    new Point3(-0.0762, -0.0762, 0)
                ), imagePoints,
                frameStaticProperties.cameraCalibration.cameraIntrinsics.getAsMatOfDouble(),
                frameStaticProperties.cameraCalibration.cameraExtrinsics.getAsMatOfDouble(),
                rvecs, tvecs,
                false, Calib3d.SOLVEPNP_IPPE_SQUARE,
                Mat.zeros(3, 1, CvType.CV_32F), Mat.zeros(3, 1, CvType.CV_32F),
                new Mat()
            );
            var bestTvec = tvecs.get(0);
            var bestRvec = rvecs.get(0);
            Translation3d translation =
                new Translation3d(bestTvec.get(0, 0)[0], bestTvec.get(1, 0)[0], bestTvec.get(2, 0)[0]);
            Rotation3d rotation =
                new Rotation3d(
                        VecBuilder.fill(bestRvec.get(0, 0)[0], bestRvec.get(1, 0)[0], bestRvec.get(2, 0)[0]),
                        Core.norm(bestRvec));
            var correctedBestPose = MathUtils.convertOpenCVtoPhotonPose(new Transform3d(translation, rotation));
            var correctedAltPose = MathUtils.convertOpenCVtoPhotonPose(new Transform3d());
            logger.debug("IPPE_SQUARE: "+(WPIUtilJNI.now() - timeStart)*1e-6);
            
            target.setBestCameraToTarget3d(
                    new Transform3d(correctedBestPose.getTranslation(), correctedBestPose.getRotation()));
            target.setAltCameraToTarget3d(
                    new Transform3d(correctedAltPose.getTranslation(), correctedAltPose.getRotation()));

            targetList.add(target);
        }

        var fpsResult = calculateFPSPipe.run(null);
        var fps = fpsResult.output;

        return new CVPipelineResult(sumPipeNanosElapsed, fps, targetList, outputFrame, inputFrame);
    }
}
