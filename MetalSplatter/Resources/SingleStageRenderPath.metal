#include "SplatProcessing.h"

vertex FragmentIn singleStageSplatVertexShader(uint vertexID [[vertex_id]],
                                               uint instanceID [[instance_id]],
                                               ushort amplificationID [[amplification_id]],
                                               device const ChunkInfo* chunks [[ buffer(BufferIndexChunks) ]],
                                               constant ChunkedSplatIndex* splatIndexArray [[ buffer(BufferIndexSplatIndex) ]],
                                               constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]],
                                               constant ObjectCropUniforms &crop [[ buffer(BufferIndexObjectCrop) ]]) {
    uint viewIndex = min(uint(amplificationID), uint(kMaxViewCount - 1));
    Uniforms uniforms = uniformsArray.uniforms[viewIndex];

    uint splatID = instanceID * uniforms.indexedSplatCount + (vertexID / 4);
    if (splatID >= uniforms.splatCount) {
        FragmentIn out;
        out.position = float4(1, 1, 0, 1);
        return out;
    }

    ChunkedSplatIndex idx = splatIndexArray[splatID];

    // Bounds check chunk index
    if (idx.chunkIndex >= uniforms.chunkCount) {
        FragmentIn out;
        out.position = float4(1, 1, 0, 1);
        return out;
    }

    ChunkInfo chunk = chunks[idx.chunkIndex];

    // Bounds check local splat index; protects against transient stale indices.
    if (idx.splatIndex >= chunk.splatCount) {
        FragmentIn out;
        out.position = float4(1, 1, 0, 1);
        return out;
    }

    if (!chunk.enabled) {
        FragmentIn out;
        out.position = float4(1, 1, 0, 1);
        return out;
    }

    Splat splat = chunk.splats[idx.splatIndex];

    return splatVertex(splat, uniforms, vertexID % 4,
                       chunk.shCoefficients, chunk.shDegree,
                       idx.splatIndex, crop, viewIndex);
}

fragment half4 singleStageSplatFragmentShader(FragmentIn in [[stage_in]],
                                              constant ObjectCropUniforms &crop [[ buffer(BufferIndexObjectCrop) ]],
                                              constant float4 *planes [[ buffer(BufferIndexObjectCropPlanes) ]]) {
    if (!splatFragmentInsideCrop(in.modelPosition, crop, planes)) {
        discard_fragment();
    }
    half alpha = splatFragmentAlpha(in.relativePosition, in.color.a);
    return half4(alpha * in.color.rgb, alpha);
}
