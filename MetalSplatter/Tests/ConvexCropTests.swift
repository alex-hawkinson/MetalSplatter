import XCTest
import Metal
import simd
@testable import MetalSplatter

/// CPU contracts only; these do not instantiate a renderer or Metal device.
/// Actual fragment discard/color/depth parity needs the separate renderer gate.
final class ConvexCropTests: XCTestCase {
    private func viewport(_ view: simd_float4x4 = matrix_identity_float4x4,
                          projection: simd_float4x4 = matrix_identity_float4x4) -> SplatRenderer.ViewportDescriptor {
        .init(viewport: MTLViewport(originX: 0, originY: 0, width: 640, height: 480,
                                    znear: 0, zfar: 1),
              projectionMatrix: projection, viewMatrix: view, screenSize: SIMD2(640, 480))
    }

    func testPlaneValidationAndFixedCoefficients() throws {
        for normal in [SIMD3<Float>.zero, SIMD3(.nan, 1, 0), SIMD3(0, .infinity, 1)] {
            XCTAssertThrowsError(try SplatRenderer.CropPlane(normal: normal, offset: 0))
        }
        XCTAssertThrowsError(try SplatRenderer.CropPlane(normal: SIMD3(1, 0, 0), offset: .infinity))
        let plane = try SplatRenderer.CropPlane(normal: SIMD3(2, -3, 4), offset: 5, strict: true)
        let crop = try SplatRenderer.ConvexCrop(planes: [plane])
        XCTAssertEqual(crop.encodedPlanes, [SIMD4(2, -3, 4, 5)])
        XCTAssertEqual(crop.strictPlaneMask, 1)
    }

    func testPlaneLimitAndTwentiethStrictBit() throws {
        let plane = try SplatRenderer.CropPlane(normal: SIMD3(1, 0, 0), offset: 0)
        let strict = try SplatRenderer.CropPlane(normal: SIMD3(0, 0, 1), offset: 0, strict: true)
        let crop = try SplatRenderer.ConvexCrop(planes: Array(repeating: plane, count: 19) + [strict])
        XCTAssertEqual(crop.encodedPlanes.count, 20)
        XCTAssertEqual(crop.strictPlaneMask, UInt32(1) << 19)
        XCTAssertThrowsError(try SplatRenderer.ConvexCrop(planes: Array(repeating: plane, count: 21)))
    }

    func testUniformABI() {
        typealias U = SplatRenderer.ObjectCropUniforms
        XCTAssertEqual(MemoryLayout<U>.stride, 144)
        XCTAssertEqual(MemoryLayout<U>.alignment, 16)
        XCTAssertEqual(MemoryLayout<U>.offset(of: \U.clipToModel0), 0)
        XCTAssertEqual(MemoryLayout<U>.offset(of: \U.clipToModel1), 64)
        XCTAssertEqual(MemoryLayout<U>.offset(of: \U.planeCount), 128)
        XCTAssertEqual(MemoryLayout<U>.offset(of: \U.strictPlaneMask), 132)
        XCTAssertEqual(MemoryLayout<U>.offset(of: \U._padding), 136)
        XCTAssertEqual(MemoryLayout<SIMD4<Float>>.stride, 16)
    }

    func testDisabledCropDoesNotInvertOrRequireViewports() throws {
        let nilUniforms = try SplatRenderer.objectCropUniforms(nil, viewports: [], maxViewCount: 0)
        let empty = try SplatRenderer.ConvexCrop(planes: [])
        let zero = simd_float4x4(columns: (.zero, .zero, .zero, .zero))
        let emptyUniforms = try SplatRenderer.objectCropUniforms(empty,
            viewports: [viewport(zero)], maxViewCount: 1)
        XCTAssertEqual(nilUniforms.planeCount, 0)
        XCTAssertEqual(emptyUniforms.planeCount, 0)
        for column in 0..<4 {
            XCTAssertEqual(emptyUniforms.clipToModel0[column], matrix_identity_float4x4[column])
        }
    }

    func testEnabledCropRejectsInvalidTransformsAndViewCount() throws {
        let plane = try SplatRenderer.CropPlane(normal: SIMD3(1, 0, 0), offset: 0)
        let crop = try SplatRenderer.ConvexCrop(planes: [plane])
        XCTAssertThrowsError(try SplatRenderer.objectCropUniforms(crop, viewports: [], maxViewCount: 2))
        XCTAssertThrowsError(try SplatRenderer.objectCropUniforms(crop,
            viewports: [viewport(), viewport()], maxViewCount: 1))
        XCTAssertThrowsError(try SplatRenderer.objectCropUniforms(crop,
            viewports: Array(repeating: viewport(), count: 3), maxViewCount: 3))
        let zero = simd_float4x4(columns: (.zero, .zero, .zero, .zero))
        XCTAssertThrowsError(try SplatRenderer.objectCropUniforms(crop,
            viewports: [viewport(zero)], maxViewCount: 1))
        var invalid = matrix_identity_float4x4
        invalid[1][2] = .nan
        XCTAssertThrowsError(try SplatRenderer.objectCropUniforms(crop,
            viewports: [viewport(invalid)], maxViewCount: 1))
        let huge = simd_float4x4(columns: (SIMD4(1e10, 0, 0, 0), SIMD4(0, 1e10, 0, 0),
                                          SIMD4(0, 0, 1e10, 0), SIMD4(0, 0, 0, 1e10)))
        XCTAssertThrowsError(try SplatRenderer.objectCropUniforms(crop,
            viewports: [viewport(huge)], maxViewCount: 1))
    }

    func testBillboardReconstructionUndoesFullObjectTransformForBothEyes() throws {
        let plane = try SplatRenderer.CropPlane(normal: SIMD3(1, 0, 0), offset: 0)
        let crop = try SplatRenderer.ConvexCrop(planes: [plane])
        let rotation = simd_float4x4(simd_quatf(angle: 0.63, axis: simd_normalize(SIMD3<Float>(1, 2, 3))))
        var scale = matrix_identity_float4x4
        scale[0][0] = 1.7; scale[1][1] = 0.8; scale[2][2] = 1.2
        var translation = matrix_identity_float4x4
        translation[3] = SIMD4(0.2, -0.1, -3, 1)
        let model = translation * rotation * scale
        // Off-axis perspective with reverse depth, matching a Metal billboard.
        let projection = simd_float4x4(columns: (
            SIMD4(1.3, 0, 0, 0), SIMD4(0, 1.7, 0, 0),
            SIMD4(0.1, -0.05, 0, -1), SIMD4(0, 0, 0.1, 0)))
        var left = matrix_identity_float4x4; left[3][0] = -0.03
        var right = matrix_identity_float4x4; right[3][0] = 0.03
        let views = [viewport(left * model, projection: projection),
                     viewport(right * model, projection: projection)]
        let uniforms = try SplatRenderer.objectCropUniforms(crop, viewports: views, maxViewCount: 2)
        let inverseMatrices = [uniforms.clipToModel0, uniforms.clipToModel1]
        for i in 0..<2 {
            let mvp = views[i].projectionMatrix * views[i].viewMatrix
            let center = mvp * SIMD4<Float>(0, 0, 0, 1)
            let recoveredCenter = inverseMatrices[i] * center
            for axis in 0..<3 { XCTAssertEqual(recoveredCenter[axis] / recoveredCenter.w, 0, accuracy: 0.00001) }
            let offsets = [SIMD2<Float>(-0.2, -0.1), SIMD2(-0.2, 0.1), SIMD2(0.2, -0.1)]
            let weights: [Float] = [0.2, 0.3, 0.5]
            var interpolated = SIMD4<Float>.zero
            var ndc = SIMD3<Float>.zero
            var reciprocalW: Float = 0
            for j in 0..<3 {
                var clip = center
                clip.x += offsets[j].x * center.w
                clip.y += offsets[j].y * center.w
                interpolated += weights[j] * (inverseMatrices[i] * clip) / clip.w
                ndc += weights[j] * SIMD3(clip.x, clip.y, clip.z) / clip.w
                reciprocalW += weights[j] / clip.w
            }
            interpolated /= reciprocalW
            let expected = inverseMatrices[i] * SIMD4(ndc, 1)
            for axis in 0..<3 {
                XCTAssertEqual(interpolated[axis] / interpolated.w,
                               expected[axis] / expected.w, accuracy: 0.00001)
            }
        }
    }
}
