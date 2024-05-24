importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');

let model = null;

self.onmessage = async (e) => {
  const { type, payload } = e.data;
  try {
    const response = await fetch('http://localhost:3000/_next/static/chunks/pages/yolov7-tiny_640x640.onnx');
    if (!model) {
      throw new Error(`Failed to fetch ${payload.url}`);
    }
    const arrayBuffer = await response.arrayBuffer();
    model = await ort.InferenceSession.create(response.url, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    self.postMessage({ type: "createModel", success: true });
  } catch (error) {
    console.error('Error loading model:', error);
    self.postMessage({ type: 'error', message: 'Error loading model' });
  }

  if (type === 'dispatchModel' && model) {
    try {
        const feeds = {};
        feeds[model.inputNames[0]] = new ort.Tensor(payload.type, payload.data, payload.dims);
        const start = Date.now();
        const outputData = await model.run(feeds);
        const end = Date.now();
        const inferenceTime = end - start;
        const output = outputData[model.outputNames[0]];
        self.postMessage({ type: "modelDispatched", output, inferenceTime });
    } catch (error) {
      console.error('Error running model:', error);
      self.postMessage({ type: "dispatchModel", success: false, error });
    }
  } else {
    self.postMessage({ type: "error", error: "Unknown message type" });
  }
};

// ./_next/static/chunks/pages/${modelName}