//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AdaINResBlock1: Module {
  @ModuleInfo var convs1: [ConvWeighted] = []
  @ModuleInfo var convs2: [ConvWeighted] = []
  @ModuleInfo var adain1: [AdaIN1d] = []
  @ModuleInfo var adain2: [AdaIN1d] = []
  var alpha1: [MLXArray] = []
  var alpha2: [MLXArray] = []

  private func getPadding(kernelSize: Int, dilation: Int = 1) -> Int {
    return Int((kernelSize * dilation - dilation) / 2)
  }

  init(
    weights: [String: MLXArray],
    weightPrefixKey: String,
    channels: Int,
    kernelSize: Int = 3,
    dilation: [Int] = [1, 3, 5],
    styleDim: Int = 64
  ) {
    let padding = { (ks: Int, d: Int) -> Int in Int((ks * d - d) / 2) }

    var localConvs1: [ConvWeighted] = []
    var localConvs2: [ConvWeighted] = []
    var localAdain1: [AdaIN1d] = []
    var localAdain2: [AdaIN1d] = []

    for i in 0 ..< 3 {
      let dilationValue = dilation[i]
      localConvs1.append(ConvWeighted(
        weightG: weights[weightPrefixKey + ".convs1.\(i).weight_g"]!,
        weightV: weights[weightPrefixKey + ".convs1.\(i).weight_v"]!,
        bias: weights[weightPrefixKey + ".convs1.\(i).bias"]!,
        stride: 1,
        padding: padding(kernelSize, dilationValue),
        dilation: dilationValue
      ))
    }

    for i in 0 ..< 3 {
      localConvs2.append(ConvWeighted(
        weightG: weights[weightPrefixKey + ".convs2.\(i).weight_g"]!,
        weightV: weights[weightPrefixKey + ".convs2.\(i).weight_v"]!,
        bias: weights[weightPrefixKey + ".convs2.\(i).bias"]!,
        stride: 1,
        padding: padding(kernelSize, 1),
        dilation: 1
      ))
    }

    for i in 0 ..< 3 {
      localAdain1.append(AdaIN1d(
        styleDim: styleDim,
        numFeatures: channels,
        fcWeight: weights[weightPrefixKey + ".adain1.\(i).fc.weight"]!,
        fcBias: weights[weightPrefixKey + ".adain1.\(i).fc.bias"]!
      ))

      localAdain2.append(AdaIN1d(
        styleDim: styleDim,
        numFeatures: channels,
        fcWeight: weights[weightPrefixKey + ".adain2.\(i).fc.weight"]!,
        fcBias: weights[weightPrefixKey + ".adain2.\(i).fc.bias"]!
      ))
    }

    convs1 = localConvs1
    convs2 = localConvs2
    adain1 = localAdain1
    adain2 = localAdain2

    super.init()

    for i in 0 ..< 3 {
      alpha1.append(weights[weightPrefixKey + ".alpha1.\(i)"]!)
      alpha2.append(weights[weightPrefixKey + ".alpha2.\(i)"]!)
    }
  }

  func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
    var result = x

    for i in 0 ..< convs1.count {
      let c1 = convs1[i]
      let c2 = convs2[i]
      let n1 = adain1[i]
      let n2 = adain2[i]
      let a1 = alpha1[i]
      let a2 = alpha2[i]

      var xt = n1(result, s: s)
      xt = xt + (1 / a1) * (MLX.sin(a1 * xt).pow(2))

      xt = MLX.swappedAxes(xt, 2, 1)
      xt = c1(xt, conv: MLX.conv1d)
      xt = MLX.swappedAxes(xt, 2, 1)

      xt = n2(xt, s: s)
      xt = xt + (1 / a2) * (MLX.sin(a2 * xt).pow(2))

      xt = MLX.swappedAxes(xt, 2, 1)
      xt = c2(xt, conv: MLX.conv1d)
      xt = MLX.swappedAxes(xt, 2, 1)

      result = xt + result
    }
    return result
  }
}
