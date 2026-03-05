//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

final class TextEncoder: Module {
  @ModuleInfo var embedding: Embedding
  @ModuleInfo var cnnConvs: [ConvWeighted]
  @ModuleInfo var cnnNorms: [LayerNormInference]
  @ModuleInfo var cnnActv: LeakyReLU
  @ModuleInfo var lstm: LSTM

  init(weights: [String: MLXArray], channels: Int, kernelSize: Int, depth: Int, nSymbols _: Int, actv: LeakyReLU = LeakyReLU(negativeSlope: 0.2)) {
    embedding = Embedding(weight: weights["text_encoder.embedding.weight"]!)

    let padding = (kernelSize - 1) / 2

    var localConvs: [ConvWeighted] = []
    var localNorms: [LayerNormInference] = []
    for i in 0 ..< depth {
      localConvs.append(
        ConvWeighted(
          weightG: weights["text_encoder.cnn.\(i).0.weight_g"]!,
          weightV: weights["text_encoder.cnn.\(i).0.weight_v"]!,
          bias: weights["text_encoder.cnn.\(i).0.bias"]!,
          padding: padding
        )
      )
      localNorms.append(
        LayerNormInference(
          weight: weights["text_encoder.cnn.\(i).1.gamma"]!,
          bias: weights["text_encoder.cnn.\(i).1.beta"]!
        )
      )
    }
    cnnConvs = localConvs
    cnnNorms = localNorms
    cnnActv = actv

    lstm = LSTM(
      inputSize: channels,
      hiddenSize: channels / 2,
      wxForward: weights["text_encoder.lstm.weight_ih_l0"]!,
      whForward: weights["text_encoder.lstm.weight_hh_l0"]!,
      biasIhForward: weights["text_encoder.lstm.bias_ih_l0"]!,
      biasHhForward: weights["text_encoder.lstm.bias_hh_l0"]!,
      wxBackward: weights["text_encoder.lstm.weight_ih_l0_reverse"]!,
      whBackward: weights["text_encoder.lstm.weight_hh_l0_reverse"]!,
      biasIhBackward: weights["text_encoder.lstm.bias_ih_l0_reverse"]!,
      biasHhBackward: weights["text_encoder.lstm.bias_hh_l0_reverse"]!
    )

    super.init()
  }

  public func callAsFunction(_ x: MLXArray, inputLengths _: MLXArray, m: MLXArray) -> MLXArray {
    var x = embedding(x)
    x = x.transposed(0, 2, 1)
    let mask = m.expandedDimensions(axis: 1)
    x = MLX.where(mask, 0.0, x)

    for i in 0 ..< cnnConvs.count {
      x = MLX.swappedAxes(x, 2, 1)
      x = cnnConvs[i](x, conv: MLX.conv1d)
      x = MLX.swappedAxes(x, 2, 1)
      x = MLX.where(mask, 0.0, x)

      x = MLX.swappedAxes(x, 2, 1)
      x = cnnNorms[i](x)
      x = MLX.swappedAxes(x, 2, 1)
      x = MLX.where(mask, 0.0, x)

      x = cnnActv(x)
      x = MLX.where(mask, 0.0, x)
    }

    x = MLX.swappedAxes(x, 2, 1)
    let (lstmOutput, _) = lstm(x)
    x = MLX.swappedAxes(lstmOutput, 2, 1)

    let xPad = MLX.zeros([x.shape[0], x.shape[1], mask.shape[mask.shape.count - 1]])
    xPad._updateInternal(x)

    return MLX.where(mask, 0.0, xPad)
  }
}
