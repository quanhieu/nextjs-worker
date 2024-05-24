import { InferenceSession, Tensor } from "onnxruntime-web";

let model: InferenceSession | null = null;

self.onmessage = async (event) => {
  const { type, payload } = event.data;
  try {    
    if (!model) {
      const response = await fetch('./pages/yolov7-tiny_640x640.onnx');
      console.log(response);
      if (!response.ok) {
        throw new Error(`Failed to fetch `);
      }

      const arrayBuffer = await response.arrayBuffer();
      model = await InferenceSession.create(arrayBuffer, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      self.postMessage({ type: "createModel", success: true });
    }
  } catch (error) {
    self.postMessage({ type: "createModel", success: false, error });
  }

  
  switch (type) {
    case "createModel":
      break;

    case "dispatchModel":
      if (model) {
        try {
          console.log('model ', model)
          const feeds: Record<string, Tensor> = {};
          feeds[model.inputNames[0]] = payload.preprocessedData;
          const start = Date.now();
          const outputData = await model.run(feeds);
          const end = Date.now();
          const inferenceTime = end - start;
          const output = outputData[model.outputNames[0]];
          self.postMessage({
            type: "dispatchModel",
            success: true,
            result: [output, inferenceTime],
          });
        } catch (error) {
          self.postMessage({ type: "dispatchModel", success: false, error });
        }
      } else {
        self.postMessage({
          type: "dispatchModel",
          success: false,
          error: "Model not initialized",
        });
      }
      break;

    default:
      self.postMessage({ type: "error", error: "Unknown message type" });
      break;
  }
};

export default null as any;

