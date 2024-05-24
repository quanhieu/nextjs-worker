importScripts('_next/static/chunks/pages/onnxruntime-web.js'); // Adjust path if necessary

let model = null;

onmessage = async (e) => {
  const { type, data } = e.data;

  if (type === 'loadModel') {
    const { url } = data;
    model = await ort.InferenceSession.create(url, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    postMessage({ type: 'modelLoaded' });
  }

  if (type === 'runModel') {
    const { preprocessedData } = data;
    const feeds = {};
    feeds[model.inputNames[0]] = new ort.Tensor('float32', preprocessedData.data, preprocessedData.dims);
    const start = Date.now();
    const outputData = await model.run(feeds);
    const end = Date.now();
    const inferenceTime = end - start;
    const output = outputData[model.outputNames[0]];
    postMessage({ type: 'modelOutput', output, inferenceTime });
  }
};