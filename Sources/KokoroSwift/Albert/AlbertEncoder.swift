//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AlbertEncoder: Module {
  let config: AlbertModelArgs
  @ModuleInfo var embeddingHiddenMappingIn: Linear
  @ModuleInfo var albertLayerGroups: [AlbertLayerGroup]

  init(weights: [String: MLXArray], config: AlbertModelArgs) {
    self.config = config
    embeddingHiddenMappingIn = Linear(weight: weights["bert.encoder.embedding_hidden_mapping_in.weight"]!,
                                      bias: weights["bert.encoder.embedding_hidden_mapping_in.bias"]!)

    var groups: [AlbertLayerGroup] = []
    for layerNum in 0 ..< config.numHiddenGroups {
      groups.append(AlbertLayerGroup(config: config, layerNum: layerNum, weights: weights))
    }
    albertLayerGroups = groups

    super.init()
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    attentionMask: MLXArray? = nil
  ) -> MLXArray {
    var output = embeddingHiddenMappingIn(hiddenStates)

    for i in 0 ..< config.numHiddenLayers {
      let groupIdx = i / (config.numHiddenLayers / config.numHiddenGroups)

      output = albertLayerGroups[groupIdx](output, attentionMask: attentionMask)
    }

    return output
  }
}
