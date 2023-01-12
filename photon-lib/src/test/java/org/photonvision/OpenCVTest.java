package org.photonvision;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Core;
import org.photonvision.util.OpenCVHelp;

import edu.wpi.first.apriltag.jni.AprilTagJNI;
import edu.wpi.first.cscore.CameraServerCvJNI;
import edu.wpi.first.cscore.CameraServerJNI;
import edu.wpi.first.hal.JNIWrapper;
import edu.wpi.first.math.MathUtil;
import edu.wpi.first.math.geometry.Pose3d;
import edu.wpi.first.math.geometry.Rotation2d;
import edu.wpi.first.math.geometry.Rotation3d;
import edu.wpi.first.math.geometry.Transform3d;
import edu.wpi.first.math.geometry.Translation3d;
import edu.wpi.first.net.WPINetJNI;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.networktables.NetworkTablesJNI;
import edu.wpi.first.util.CombinedRuntimeLoader;
import edu.wpi.first.util.RuntimeLoader;
import edu.wpi.first.util.WPIUtilJNI;

public class OpenCVTest {
    private static final double kTrlDelta = 0.005;
    private static final double kRotDeltaDeg = 0.25;

    public static void assertSame(Translation3d trl1, Translation3d trl2) {
        assertEquals(0, trl1.getX()-trl2.getX(), kTrlDelta, "Trl X Diff");
        assertEquals(0, trl1.getY()-trl2.getY(), kTrlDelta, "Trl Y Diff");
        assertEquals(0, trl1.getZ()-trl2.getZ(), kTrlDelta, "Trl Z Diff");
    }
    public static void assertSame(Rotation3d rot1, Rotation3d rot2) {
        assertEquals(0, MathUtil.angleModulus(rot1.getX()-rot2.getX()), kRotDeltaDeg, "Rot X Diff");
        assertEquals(0, MathUtil.angleModulus(rot1.getY()-rot2.getY()), kRotDeltaDeg, "Rot Y Diff");
        assertEquals(0, MathUtil.angleModulus(rot1.getZ()-rot2.getZ()), kRotDeltaDeg, "Rot Z Diff");
        assertEquals(0, MathUtil.angleModulus(rot1.getAngle()-rot2.getAngle()), kRotDeltaDeg, "Rot W Diff");
    }
    public static void assertSame(Pose3d pose1, Pose3d pose2) {
        assertSame(pose1.getTranslation(), pose2.getTranslation());
        assertSame(pose1.getRotation(), pose2.getRotation());
    }
    public static void assertSame(Transform3d trf1, Transform3d trf2) {
        assertSame(trf1.getTranslation(), trf2.getTranslation());
        assertSame(trf1.getRotation(), trf2.getRotation());
    }

    private static final CameraProperties prop = new CameraProperties();

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
                "cscorejnicvstatic"
            );

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

    @Test
    public void testTrlConvert() {
        var trl = new Translation3d(0.75, -0.4, 0.1);
        var tvec = OpenCVHelp.translationToTvec(trl);
        var result = OpenCVHelp.tvecToTranslation(tvec);
        tvec.release();
        assertSame(trl, result);
    }
    @Test
    public void testRotConvert() {
        var rot = new Rotation3d(0.5, 1, -1);
        var rvec = OpenCVHelp.rotationToRvec(rot);
        var result = OpenCVHelp.rvecToRotation(rvec);
        rvec.release();
        var diff = result.minus(rot);
        assertSame(new Rotation3d(), diff);
    }
    @Test
    public void testProjection() {
        var target = new VisionTargetSim(
            new Pose3d(1, 0, 0, new Rotation3d(0, 0, Math.PI)),
            TargetModel.kTag16h5,
            0
        );
        var cameraPose = new Pose3d(0, 0, 0, new Rotation3d());
        var targetCorners = OpenCVHelp.projectPoints(
            prop,
            cameraPose,
            target.getFieldVertices()
        );
        // find circulation (counter/clockwise-ness)
        double circulation = 0;
        for(int i=0; i<targetCorners.size(); i++) {
            double xDiff = targetCorners.get((i+1)%4).x - targetCorners.get(i).x;
            double ySum = targetCorners.get((i+1)%4).y + targetCorners.get(i).y;
            circulation += xDiff * ySum;
        }
        assertTrue(circulation > 0, "2d fiducial points aren't counter-clockwise");
        // undo projection distortion
        targetCorners = prop.undistort(targetCorners);
        var avgCenterRot1 = prop.getPixelRot(OpenCVHelp.averageCorner(targetCorners));
        cameraPose = cameraPose.plus(new Transform3d(new Translation3d(), new Rotation3d(0, 0.25, 0.25)));
        targetCorners = OpenCVHelp.projectPoints(
            prop,
            cameraPose,
            target.getFieldVertices()
        );
        var avgCenterRot2 = prop.getPixelRot(OpenCVHelp.averageCorner(targetCorners));
        var yaw2d = new Rotation2d(avgCenterRot2.getZ());
        var pitch2d = new Rotation2d(avgCenterRot2.getY());
        var yawDiff = yaw2d.minus(new Rotation2d(avgCenterRot1.getZ()));
        var pitchDiff = pitch2d.minus(new Rotation2d(avgCenterRot1.getY()));
        assertTrue(yawDiff.getRadians() < 0, "2d points don't follow yaw");
        assertTrue(pitchDiff.getRadians() < 0, "2d points don't follow pitch");
        var actualRelation = new CameraTargetRelation(cameraPose, target.getPose());
        assertEquals(
            actualRelation.camToTargPitch.getDegrees(),
            pitchDiff.getDegrees() * Math.cos(yaw2d.getRadians()), // adjust for perpsective distortion
            kRotDeltaDeg,
            "2d pitch doesn't match 3d"
        );
        assertEquals(
            actualRelation.camToTargYaw.getDegrees(),
            yawDiff.getDegrees(),
            kRotDeltaDeg,
            "2d yaw doesn't match 3d"
        );
    }
    @Test
    public void testSolvePNP_SQUARE() {
        var target = new VisionTargetSim(
            new Pose3d(5, 0.5, 1, new Rotation3d(0, 0, Math.PI)),
            TargetModel.kTag16h5,
            0
        );
        var cameraPose = new Pose3d(0, 0, 0, new Rotation3d());
        var actualRelation = new CameraTargetRelation(cameraPose, target.getPose());
        var targetCorners = OpenCVHelp.projectPoints(
            prop,
            cameraPose,
            target.getFieldVertices()
        );
        var pnpSim = OpenCVHelp.solvePNP_SQUARE(prop, target.getModel().vertices, targetCorners);
        var estRelation = new CameraTargetRelation(
            cameraPose,
            cameraPose.plus(pnpSim.best)
        );
        assertSame(actualRelation.camToTarg, estRelation.camToTarg);
    }
    @Test
    public void testSolvePNP_SQPNP() {
        var target = new VisionTargetSim(
            new Pose3d(5, 0.5, 1, new Rotation3d(0, 0, Math.PI)),
            new TargetModel(
                List.of(
                    new Translation3d(0, 0, 0),
                    new Translation3d(1, 0, 0),
                    new Translation3d(0, 1, 0),
                    new Translation3d(0, 0, 1),
                    new Translation3d(0.125, 0.25, 0.5),
                    new Translation3d(0, 0, -1),
                    new Translation3d(0, -1, 0),
                    new Translation3d(-1, 0, 0)
                )
            ),
            0
        );
        var cameraPose = new Pose3d(0, 0, 0, new Rotation3d());
        var actualRelation = new CameraTargetRelation(cameraPose, target.getPose());
        var targetCorners = OpenCVHelp.projectPoints(
            prop,
            cameraPose,
            target.getFieldVertices()
        );
        var pnpSim = OpenCVHelp.solvePNP_SQPNP(prop, target.getModel().vertices, targetCorners);
        var estRelation = new CameraTargetRelation(
            cameraPose,
            cameraPose.plus(pnpSim.best)
        );
        assertSame(actualRelation.camToTarg, estRelation.camToTarg);
    }
}
