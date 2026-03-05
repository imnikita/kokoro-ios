//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class Generator: Module {
  let numKernels: Int
  let numUpsamples: Int
  @ModuleInfo var mSource: SourceModuleHnNSF
  @ModuleInfo var f0Upsample: Upsample
  let postNFFt: Int
  var noiseConvs: [Conv1dInference]
  @ModuleInfo var noiseRes: [AdaINResBlock1]
  @ModuleInfo var ups: [ConvWeighted]
  @ModuleInfo var resBlocks: [AdaINResBlock1]
  @ModuleInfo var convPost: ConvWeighted
  @ModuleInfo var reflectionPad: ReflectionPad1d
  let stft: MLXSTFT

  init(weights: [String: MLXArray],
       styleDim: Int,
       resblockKernelSizes: [Int],
       upsampleRates: [Int],
       upsampleInitialChannel: Int,
       resblockDilationSizes: [[Int]],
       upsampleKernelSizes: [Int],
       genIstftNFft: Int,
       genIstftHopSize: Int)
  {
    numKernels = resblockKernelSizes.count
    numUpsamples = upsampleRates.count

    let upsampleScaleNum = MLX.product(MLXArray(upsampleRates)) * genIstftHopSize
    let upsampleScaleNumVal: Int = upsampleScaleNum.item()

    mSource = SourceModuleHnNSF(
      weights: weights,
      samplingRate: KokoroTTS.Constants.samplingRate,
      upsampleScale: upsampleScaleNum.item(),
      harmonicNum: 8,
      voicedThreshold: 10
    )

    f0Upsample = Upsample(scaleFactor: .float(Float(upsampleScaleNumVal)))

    var localNoiseConvs: [Conv1dInference] = []
    var localNoiseRes: [AdaINResBlock1] = []
    var localUps: [ConvWeighted] = []
    var localResBlocks: [AdaINResBlock1] = []

    for (i, (u, k)) in zip(upsampleRates, upsampleKernelSizes).enumerated() {
      localUps.append(
        ConvWeighted(
          weightG: weights["decoder.generator.ups.\(i).weight_g"]!,
          weightV: weights["decoder.generator.ups.\(i).weight_v"]!,
          bias: weights["decoder.generator.ups.\(i).bias"]!,
          stride: u,
          padding: (k - u) / 2
        )
      )
    }

    for i in 0 ..< localUps.count {
      let ch = upsampleInitialChannel / Int(pow(2.0, Double(i + 1)))
      for (j, (k, d)) in zip(resblockKernelSizes, resblockDilationSizes).enumerated() {
        localResBlocks.append(
          AdaINResBlock1(
            weights: weights,
            weightPrefixKey: "decoder.generator.resblocks.\((i * resblockKernelSizes.count) + j)",
            channels: ch,
            kernelSize: k,
            dilation: d,
            styleDim: styleDim
          )
        )
      }

      let cCur = ch
      if i + 1 < upsampleRates.count {
        let strideF0: Int = MLX.product(MLXArray(upsampleRates)[(i + 1)...]).item()
        localNoiseConvs.append(
          Conv1dInference(
            inputChannels: genIstftNFft + 2,
            outputChannels: cCur,
            kernelSize: strideF0 * 2,
            stride: strideF0,
            padding: (strideF0 + 1) / 2,
            weight: weights["decoder.generator.noise_convs.\(i).weight"]!,
            bias: weights["decoder.generator.noise_convs.\(i).bias"]!
          )
        )

        localNoiseRes.append(
          AdaINResBlock1(
            weights: weights,
            weightPrefixKey: "decoder.generator.noise_res.\(i)",
            channels: cCur,
            kernelSize: 7,
            dilation: [1, 3, 5],
            styleDim: styleDim
          )
        )
      } else {
        localNoiseConvs.append(
          Conv1dInference(
            inputChannels: genIstftNFft + 2,
            outputChannels: cCur,
            kernelSize: 1,
            weight: weights["decoder.generator.noise_convs.\(i).weight"]!,
            bias: weights["decoder.generator.noise_convs.\(i).bias"]!
          )
        )
        localNoiseRes.append(
          AdaINResBlock1(
            weights: weights,
            weightPrefixKey: "decoder.generator.noise_res.\(i)",
            channels: cCur,
            kernelSize: 11,
            dilation: [1, 3, 5],
            styleDim: styleDim
          )
        )
      }
    }

    noiseConvs = localNoiseConvs
    noiseRes = localNoiseRes
    ups = localUps
    resBlocks = localResBlocks

    postNFFt = genIstftNFft

    convPost = ConvWeighted(
      weightG: weights["decoder.generator.conv_post.weight_g"]!,
      weightV: weights["decoder.generator.conv_post.weight_v"]!,
      bias: weights["decoder.generator.conv_post.bias"]!,
      stride: 1,
      padding: 3
    )

    reflectionPad = ReflectionPad1d(padding: (1, 0))

    stft = MLXSTFT(
      filterLength: genIstftNFft,
      hopLength: genIstftHopSize,
      winLength: genIstftNFft
    )

    super.init()
  }

  func callAsFunction(_ x: MLXArray, _ s: MLXArray, _ F0Curve: MLXArray) -> MLXArray {
    var f0New = F0Curve[.newAxis, 0..., 0...].transposed(0, 2, 1)
    f0New = f0Upsample(f0New)

    var (harSource, _, _) = mSource(f0New)

    harSource = MLX.squeezed(harSource.transposed(0, 2, 1), axis: 1)
    let (harSpec, harPhase) = stft.transform(inputData: harSource)
    
    var har = MLX.concatenated([harSpec, harPhase], axis: 1)
    har = MLX.swappedAxes(har, 2, 1)
        
    var newX = x
    for i in 0 ..< numUpsamples {
      newX = LeakyReLU(negativeSlope: 0.1)(newX)
      var xSource = noiseConvs[i](har)
      xSource = MLX.swappedAxes(xSource, 2, 1)
      xSource = noiseRes[i](xSource, s)

      newX = MLX.swappedAxes(newX, 2, 1)
      newX = ups[i](newX, conv: MLX.convTransposed1d)
      newX = MLX.swappedAxes(newX, 2, 1)

      if i == numUpsamples - 1 {
        newX = reflectionPad(newX)
      }
      newX = newX + xSource
      
      var xs: MLXArray?
      for j in 0 ..< numKernels {
        if xs == nil {
          xs = resBlocks[i * numKernels + j](newX, s)
        } else {
          let temp = resBlocks[i * numKernels + j](newX, s)
          xs = xs! + temp
        }
      }
      newX = xs! / numKernels
    }
    
    newX = LeakyReLU(negativeSlope: 0.01)(newX)

    newX = MLX.swappedAxes(newX, 2, 1)
    newX = convPost(newX, conv: MLX.conv1d)
    newX = MLX.swappedAxes(newX, 2, 1)
    
    let spec = MLX.exp(newX[0..., 0 ..< (postNFFt / 2 + 1), 0...])
    let phase = MLX.sin(newX[0..., (postNFFt / 2 + 1)..., 0...])

    let result = stft.inverse(magnitude: spec, phase: phase)
    return result
  }
}
