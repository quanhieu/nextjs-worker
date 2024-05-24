importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js");

let model = null;

self.onmessage = async(event) => {
    const input = event.data;
    const output = await run_model(input);
    postMessage(output);
}

async function run_model(input) {
    if (!model) {
        model = await ort.InferenceSession.create("./models/yolov8n.onnx");
    }
    input = new ort.Tensor(Float32Array.from(input),[1, 3, 640, 640]);
    const outputs = await model.run({images:input});
    return outputs["output0"].data;
}

// you are a solution architect, have experient in nextjs and webworker, I have a function "createModelCpu" and "runModel" it very heavy and made website lag, now i need you help migrate runModel work on web-worker