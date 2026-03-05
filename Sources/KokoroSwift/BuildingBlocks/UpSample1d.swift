//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class UpSample1d: Module {
  private let layerType: String
  private var interpolate: Upsample

  init(layerType: String) {
    self.layerType = layerType
    interpolate = Upsample(
      scaleFactor: 2.0,
      mode: .nearest
    )

    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    if layerType == "none" {
      return x
    } else {
      return interpolate(x)
    }
  }
}
