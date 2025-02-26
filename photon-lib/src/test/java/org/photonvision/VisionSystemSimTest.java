/*
 * MIT License
 *
 * Copyright (c) PhotonVision
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package org.photonvision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import edu.wpi.first.apriltag.jni.AprilTagJNI;
import edu.wpi.first.cscore.CameraServerCvJNI;
import edu.wpi.first.cscore.CameraServerJNI;
import edu.wpi.first.hal.JNIWrapper;
import edu.wpi.first.math.geometry.Pose2d;
import edu.wpi.first.math.geometry.Pose3d;
import edu.wpi.first.math.geometry.Rotation2d;
import edu.wpi.first.math.geometry.Rotation3d;
import edu.wpi.first.math.geometry.Transform3d;
import edu.wpi.first.math.geometry.Translation2d;
import edu.wpi.first.math.geometry.Translation3d;
import edu.wpi.first.math.util.Units;
import edu.wpi.first.net.WPINetJNI;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.networktables.NetworkTablesJNI;
import edu.wpi.first.util.CombinedRuntimeLoader;
import edu.wpi.first.util.RuntimeLoader;
import edu.wpi.first.util.WPIUtilJNI;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.opencv.core.Core;
import org.photonvision.estimation.TargetModel;
import org.photonvision.simulation.PhotonCameraSim;
import org.photonvision.simulation.VisionSystemSim;
import org.photonvision.simulation.VisionTargetSim;
import org.photonvision.targeting.PhotonTrackedTarget;

class VisionSystemSimTest {
    private static final double kTrlDelta = 0.005;
    private static final double kRotDeltaDeg = 0.25;

    @Test
    public void testEmpty() {
        Assertions.assertDoesNotThrow(
                () -> {
                    var sysUnderTest = new VisionSystemSim("Test");
                    sysUnderTest.addVisionTargets(
                            new VisionTargetSim(new Pose3d(), new TargetModel(1.0, 1.0)));
                    for (int loopIdx = 0; loopIdx < 100; loopIdx++) {
                        sysUnderTest.update(new Pose2d());
                    }
                });
    }

    @BeforeAll
    public static void setUp() {
        JNIWrapper.Helper.setExtractOnStaticLoad(false);
        WPIUtilJNI.Helper.setExtractOnStaticLoad(false);
        NetworkTablesJNI.Helper.setExtractOnStaticLoad(false);
        WPINetJNI.Helper.setExtractOnStaticLoad(false);
        CameraServerJNI.Helper.setExtractOnStaticLoad(false);
        CameraServerCvJNI.Helper.setExtractOnStaticLoad(false);
        AprilTagJNI.Helper.setExtractOnStaticLoad(false);

        try {
            CombinedRuntimeLoader.loadLibraries(
                    VisionSystemSim.class,
                    "wpiutiljni",
                    "ntcorejni",
                    "wpinetjni",
                    "wpiHaljni",
                    "cscorejni",
                    "cscorejnicvstatic");

            var loader =
                    new RuntimeLoader<>(
                            Core.NATIVE_LIBRARY_NAME, RuntimeLoader.getDefaultExtractionRoot(), Core.class);
            loader.loadLibrary();
        } catch (Exception e) {
            e.printStackTrace();
        }

        // NT live for debug purposes
        NetworkTableInstance.getDefault().startServer();

        // No version check for testing
        PhotonCamera.setVersionCheckEnabled(false);
    }

    @AfterAll
    public static void shutDown() {}

    // @ParameterizedTest
    // @ValueSource(doubles = {5, 10, 15, 20, 25, 30})
    // public void testDistanceAligned(double dist) {
    //     final var targetPose = new Pose2d(new Translation2d(15.98, 0), new Rotation2d());
    //     var sysUnderTest =
    //             new SimVisionSystem("Test", 80.0, 0.0, new Transform2d(), 1, 99999, 320, 240, 0);
    //     sysUnderTest.addSimVisionTarget(new SimVisionTarget(targetPose, 0.0, 1.0, 1.0));

    //     final var robotPose = new Pose2d(new Translation2d(35 - dist, 0), new Rotation2d());
    //     sysUnderTest.processFrame(robotPose);

    //     var result = sysUnderTest.cam.getLatestResult();

    //     assertTrue(result.hasTargets());
    //     assertEquals(result.getBestTarget().getCameraToTarget().getTranslation().getNorm(), dist);
    // }

    @Test
    public void testVisibilityCupidShuffle() {
        final var targetPose =
                new Pose3d(new Translation3d(15.98, 0, 2), new Rotation3d(0, 0, Math.PI));
        var visionSysSim = new VisionSystemSim("Test");
        var camera = new PhotonCamera("camera");
        var cameraSim = new PhotonCameraSim(camera);
        visionSysSim.addCamera(cameraSim, new Transform3d());
        cameraSim.prop.setCalibration(640, 480, Rotation2d.fromDegrees(80));
        visionSysSim.addVisionTargets(new VisionTargetSim(targetPose, new TargetModel(1.0, 3.0), 3));

        // To the right, to the right
        var robotPose = new Pose2d(new Translation2d(5, 0), Rotation2d.fromDegrees(-70));
        visionSysSim.update(robotPose);
        assertFalse(camera.getLatestResult().hasTargets());

        // To the right, to the right
        robotPose = new Pose2d(new Translation2d(5, 0), Rotation2d.fromDegrees(-95));
        visionSysSim.update(robotPose);
        assertFalse(camera.getLatestResult().hasTargets());

        // To the left, to the left
        robotPose = new Pose2d(new Translation2d(5, 0), Rotation2d.fromDegrees(90));
        visionSysSim.update(robotPose);
        assertFalse(camera.getLatestResult().hasTargets());

        // To the left, to the left
        robotPose = new Pose2d(new Translation2d(5, 0), Rotation2d.fromDegrees(65));
        visionSysSim.update(robotPose);
        assertFalse(camera.getLatestResult().hasTargets());

        // now kick, now kick
        robotPose = new Pose2d(new Translation2d(2, 0), Rotation2d.fromDegrees(5));
        visionSysSim.update(robotPose);
        assertTrue(camera.getLatestResult().hasTargets());

        // now kick, now kick
        robotPose = new Pose2d(new Translation2d(2, 0), Rotation2d.fromDegrees(-5));
        visionSysSim.update(robotPose);
        assertTrue(camera.getLatestResult().hasTargets());

        // now walk it by yourself
        robotPose = new Pose2d(new Translation2d(2, 0), Rotation2d.fromDegrees(-179));
        visionSysSim.update(robotPose);
        assertFalse(camera.getLatestResult().hasTargets());

        // now walk it by yourself
        visionSysSim.adjustCamera(
                cameraSim, new Transform3d(new Translation3d(), new Rotation3d(0, 0, Math.PI)));
        visionSysSim.update(robotPose);
        assertTrue(camera.getLatestResult().hasTargets());
    }

    @Test
    public void testNotVisibleVert1() {
        final var targetPose =
                new Pose3d(new Translation3d(15.98, 0, 1), new Rotation3d(0, 0, Math.PI));
        var visionSysSim = new VisionSystemSim("Test");
        var camera = new PhotonCamera("camera");
        var cameraSim = new PhotonCameraSim(camera);
        visionSysSim.addCamera(cameraSim, new Transform3d());
        cameraSim.prop.setCalibration(640, 480, Rotation2d.fromDegrees(80));
        visionSysSim.addVisionTargets(new VisionTargetSim(targetPose, new TargetModel(1.0, 3.0), 3));

        var robotPose = new Pose2d(new Translation2d(5, 0), Rotation2d.fromDegrees(5));
        visionSysSim.update(robotPose);
        assertTrue(camera.getLatestResult().hasTargets());

        visionSysSim.adjustCamera( // vooop selfie stick
                cameraSim, new Transform3d(new Translation3d(0, 0, 5000), new Rotation3d(0, 0, Math.PI)));
        visionSysSim.update(robotPose);
        assertFalse(camera.getLatestResult().hasTargets());
    }

    @Test
    public void testNotVisibleVert2() {
        final var targetPose =
                new Pose3d(new Translation3d(15.98, 0, 2), new Rotation3d(0, 0, Math.PI));
        var robotToCamera =
                new Transform3d(new Translation3d(0, 0, 1), new Rotation3d(0, -Math.PI / 4, 0));
        var visionSysSim = new VisionSystemSim("Test");
        var camera = new PhotonCamera("camera");
        var cameraSim = new PhotonCameraSim(camera);
        visionSysSim.addCamera(cameraSim, robotToCamera);
        cameraSim.prop.setCalibration(1234, 1234, Rotation2d.fromDegrees(80));
        visionSysSim.addVisionTargets(new VisionTargetSim(targetPose, new TargetModel(1.0, 0.5), 1736));

        var robotPose = new Pose2d(new Translation2d(13.98, 0), Rotation2d.fromDegrees(5));
        visionSysSim.update(robotPose);
        assertTrue(camera.getLatestResult().hasTargets());

        // Pitched back camera should mean target goes out of view below the robot as distance increases
        robotPose = new Pose2d(new Translation2d(0, 0), Rotation2d.fromDegrees(5));
        visionSysSim.update(robotPose);
        assertFalse(camera.getLatestResult().hasTargets());
    }

    @Test
    public void testNotVisibleTgtSize() {
        final var targetPose =
                new Pose3d(new Translation3d(15.98, 0, 1), new Rotation3d(0, 0, Math.PI));
        var visionSysSim = new VisionSystemSim("Test");
        var camera = new PhotonCamera("camera");
        var cameraSim = new PhotonCameraSim(camera);
        visionSysSim.addCamera(cameraSim, new Transform3d());
        cameraSim.prop.setCalibration(640, 480, Rotation2d.fromDegrees(80));
        cameraSim.setMinTargetAreaPixels(20.0);
        visionSysSim.addVisionTargets(new VisionTargetSim(targetPose, new TargetModel(0.1, 0.025), 24));

        var robotPose = new Pose2d(new Translation2d(12, 0), Rotation2d.fromDegrees(5));
        visionSysSim.update(robotPose);
        assertTrue(camera.getLatestResult().hasTargets());

        robotPose = new Pose2d(new Translation2d(0, 0), Rotation2d.fromDegrees(5));
        visionSysSim.update(robotPose);
        assertFalse(camera.getLatestResult().hasTargets());
    }

    @Test
    public void testNotVisibleTooFarForLEDs() {
        final var targetPose =
                new Pose3d(new Translation3d(15.98, 0, 1), new Rotation3d(0, 0, Math.PI));
        var visionSysSim = new VisionSystemSim("Test");
        var camera = new PhotonCamera("camera");
        var cameraSim = new PhotonCameraSim(camera);
        visionSysSim.addCamera(cameraSim, new Transform3d());
        cameraSim.prop.setCalibration(640, 480, Rotation2d.fromDegrees(80));
        cameraSim.setMaxSightRange(10);
        cameraSim.setMinTargetAreaPixels(1.0);
        visionSysSim.addVisionTargets(new VisionTargetSim(targetPose, new TargetModel(1.0, 0.25), 78));

        var robotPose = new Pose2d(new Translation2d(10, 0), Rotation2d.fromDegrees(5));
        visionSysSim.update(robotPose);
        assertTrue(camera.getLatestResult().hasTargets());

        robotPose = new Pose2d(new Translation2d(0, 0), Rotation2d.fromDegrees(5));
        visionSysSim.update(robotPose);
        assertFalse(camera.getLatestResult().hasTargets());
    }

    @ParameterizedTest
    @ValueSource(doubles = {-10, -5, -0, -1, -2, 5, 7, 10.23})
    public void testYawAngles(double testYaw) {
        final var targetPose =
                new Pose3d(new Translation3d(15.98, 0, 1), new Rotation3d(0, 0, 3 * Math.PI / 4));
        var visionSysSim = new VisionSystemSim("Test");
        var camera = new PhotonCamera("camera");
        var cameraSim = new PhotonCameraSim(camera);
        visionSysSim.addCamera(cameraSim, new Transform3d());
        cameraSim.prop.setCalibration(640, 480, Rotation2d.fromDegrees(80));
        cameraSim.setMinTargetAreaPixels(0.0);
        visionSysSim.addVisionTargets(new VisionTargetSim(targetPose, new TargetModel(0.5, 0.5), 3));

        var robotPose = new Pose2d(new Translation2d(10, 0), Rotation2d.fromDegrees(-1.0 * testYaw));
        visionSysSim.update(robotPose);
        var res = camera.getLatestResult();
        assertTrue(res.hasTargets());
        var tgt = res.getBestTarget();
        assertEquals(tgt.getYaw(), testYaw, kRotDeltaDeg);
    }

    @ParameterizedTest
    @ValueSource(doubles = {-10, -5, -0, -1, -2, 5, 7, 10.23, 20.21, -19.999})
    public void testCameraPitch(double testPitch) {
        final var targetPose =
                new Pose3d(new Translation3d(15.98, 0, 0), new Rotation3d(0, 0, 3 * Math.PI / 4));
        final var robotPose = new Pose2d(new Translation2d(10, 0), new Rotation2d(0));
        var visionSysSim = new VisionSystemSim("Test");
        var camera = new PhotonCamera("camera");
        var cameraSim = new PhotonCameraSim(camera);
        visionSysSim.addCamera(cameraSim, new Transform3d());
        cameraSim.prop.setCalibration(640, 480, Rotation2d.fromDegrees(120));
        cameraSim.setMinTargetAreaPixels(0.0);
        visionSysSim.addVisionTargets(new VisionTargetSim(targetPose, new TargetModel(0.5, 0.5), 23));

        // Transform is now robot -> camera
        visionSysSim.adjustCamera(
                cameraSim,
                new Transform3d(
                        new Translation3d(), new Rotation3d(0, Units.degreesToRadians(testPitch), 0)));
        visionSysSim.update(robotPose);
        var res = camera.getLatestResult();
        assertTrue(res.hasTargets());
        var tgt = res.getBestTarget();

        // Since the camera is level with the target, a positive-upward point will mean the target is in
        // the
        // lower half of the image
        // which should produce negative pitch.
        assertEquals(testPitch, tgt.getPitch(), kRotDeltaDeg);
    }

    private static Stream<Arguments> distCalCParamProvider() {
        // Arbitrary and fairly random assortment of distances, camera pitches, and heights
        return Stream.of(
                Arguments.of(5, -15.98, 0),
                Arguments.of(6, -15.98, 1),
                Arguments.of(10, -15.98, 0),
                Arguments.of(15, -15.98, 2),
                Arguments.of(19.95, -15.98, 0),
                Arguments.of(20, -15.98, 0),
                Arguments.of(5, -42, 1),
                Arguments.of(6, -42, 0),
                Arguments.of(10, -42, 2),
                Arguments.of(15, -42, 0.5),
                Arguments.of(19.42, -15.98, 0),
                Arguments.of(20, -42, 0),
                Arguments.of(5, -35, 2),
                Arguments.of(6, -35, 0),
                Arguments.of(10, -34, 3.2),
                Arguments.of(15, -33, 0),
                Arguments.of(19.52, -15.98, 1.1));
    }

    @ParameterizedTest
    @MethodSource("distCalCParamProvider")
    public void testDistanceCalc(double testDist, double testPitch, double testHeight) {
        // Assume dist along ground and tgt height the same. Iterate over other parameters.

        final var targetPose =
                new Pose3d(new Translation3d(15.98, 0, 1), new Rotation3d(0, 0, Math.PI * 0.98));
        final var robotPose =
                new Pose3d(new Translation3d(15.98 - Units.feetToMeters(testDist), 0, 0), new Rotation3d());
        final var robotToCamera =
                new Transform3d(
                        new Translation3d(0, 0, Units.feetToMeters(testHeight)),
                        new Rotation3d(0, Units.degreesToRadians(testPitch), 0));

        var visionSysSim =
                new VisionSystemSim(
                        "absurdlylongnamewhichshouldneveractuallyhappenbuteehwelltestitanywaysohowsyourdaygoingihopegoodhaveagreatrestofyourlife!");
        var camera = new PhotonCamera("camera");
        var cameraSim = new PhotonCameraSim(camera);
        visionSysSim.addCamera(cameraSim, new Transform3d());
        cameraSim.prop.setCalibration(640, 480, Rotation2d.fromDegrees(160));
        cameraSim.setMinTargetAreaPixels(0.0);
        visionSysSim.adjustCamera(cameraSim, robotToCamera);
        // note that non-fiducial targets have different center point calculation and will
        // return slightly inaccurate yaw/pitch values
        visionSysSim.addVisionTargets(new VisionTargetSim(targetPose, new TargetModel(0.5, 0.5), 0));

        visionSysSim.update(robotPose);
        var res = camera.getLatestResult();
        assertTrue(res.hasTargets());
        var tgt = res.getBestTarget();
        assertEquals(0.0, tgt.getYaw(), kRotDeltaDeg);

        double distMeas =
                PhotonUtils.calculateDistanceToTargetMeters(
                        robotToCamera.getZ(),
                        targetPose.getZ(),
                        Units.degreesToRadians(-testPitch),
                        Units.degreesToRadians(tgt.getPitch()));
        assertEquals(Units.feetToMeters(testDist), distMeas, kTrlDelta);
    }

    @Test
    public void testMultipleTargets() {
        final var targetPoseL =
                new Pose3d(new Translation3d(15.98, 2, 0), new Rotation3d(0, 0, Math.PI));
        final var targetPoseC =
                new Pose3d(new Translation3d(15.98, 0, 0), new Rotation3d(0, 0, Math.PI));
        final var targetPoseR =
                new Pose3d(new Translation3d(15.98, -2, 0), new Rotation3d(0, 0, Math.PI));

        var visionSysSim = new VisionSystemSim("Test");
        var camera = new PhotonCamera("camera");
        var cameraSim = new PhotonCameraSim(camera);
        visionSysSim.addCamera(cameraSim, new Transform3d());
        cameraSim.prop.setCalibration(640, 480, Rotation2d.fromDegrees(80));
        cameraSim.setMinTargetAreaPixels(20.0);

        visionSysSim.addVisionTargets(
                new VisionTargetSim(
                        targetPoseL.transformBy(
                                new Transform3d(new Translation3d(0, 0, 0.00), new Rotation3d())),
                        TargetModel.kTag16h5,
                        1));
        visionSysSim.addVisionTargets(
                new VisionTargetSim(
                        targetPoseC.transformBy(
                                new Transform3d(new Translation3d(0, 0, 0.00), new Rotation3d())),
                        TargetModel.kTag16h5,
                        2));
        visionSysSim.addVisionTargets(
                new VisionTargetSim(
                        targetPoseR.transformBy(
                                new Transform3d(new Translation3d(0, 0, 0.00), new Rotation3d())),
                        TargetModel.kTag16h5,
                        3));
        visionSysSim.addVisionTargets(
                new VisionTargetSim(
                        targetPoseL.transformBy(
                                new Transform3d(new Translation3d(0, 0, 1.00), new Rotation3d())),
                        TargetModel.kTag16h5,
                        4));
        visionSysSim.addVisionTargets(
                new VisionTargetSim(
                        targetPoseC.transformBy(
                                new Transform3d(new Translation3d(0, 0, 1.00), new Rotation3d())),
                        TargetModel.kTag16h5,
                        5));
        visionSysSim.addVisionTargets(
                new VisionTargetSim(
                        targetPoseL.transformBy(
                                new Transform3d(new Translation3d(0, 0, 1.00), new Rotation3d())),
                        TargetModel.kTag16h5,
                        6));
        visionSysSim.addVisionTargets(
                new VisionTargetSim(
                        targetPoseL.transformBy(
                                new Transform3d(new Translation3d(0, 0, 0.50), new Rotation3d())),
                        TargetModel.kTag16h5,
                        7));
        visionSysSim.addVisionTargets(
                new VisionTargetSim(
                        targetPoseC.transformBy(
                                new Transform3d(new Translation3d(0, 0, 0.50), new Rotation3d())),
                        TargetModel.kTag16h5,
                        8));
        visionSysSim.addVisionTargets(
                new VisionTargetSim(
                        targetPoseL.transformBy(
                                new Transform3d(new Translation3d(0, 0, 0.75), new Rotation3d())),
                        TargetModel.kTag16h5,
                        9));
        visionSysSim.addVisionTargets(
                new VisionTargetSim(
                        targetPoseR.transformBy(
                                new Transform3d(new Translation3d(0, 0, 0.75), new Rotation3d())),
                        TargetModel.kTag16h5,
                        10));
        visionSysSim.addVisionTargets(
                new VisionTargetSim(
                        targetPoseL.transformBy(
                                new Transform3d(new Translation3d(0, 0, 0.25), new Rotation3d())),
                        TargetModel.kTag16h5,
                        11));

        var robotPose = new Pose2d(new Translation2d(6.0, 0), Rotation2d.fromDegrees(0.25));
        visionSysSim.update(robotPose);
        var res = camera.getLatestResult();
        assertTrue(res.hasTargets());
        List<PhotonTrackedTarget> tgtList;
        tgtList = res.getTargets();
        assertEquals(11, tgtList.size());
    }
}
