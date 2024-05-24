import { Tensor } from "onnxruntime-web";

export async function createModelCpu(url: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./modelWorker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = (event) => {
      const { type, success, error } = event.data;
      if (type === "createModel") {
        worker.terminate();
        success ? resolve() : reject(error);
      }
    };
    worker.postMessage({ type: "createModel", payload: { url } });
  });
}

export async function dispatchModel(preprocessedData: Tensor): Promise<[Tensor, number]> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./modelWorker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = (event) => {
      const { type, success, result, error } = event.data;
      if (type === "dispatchModel") {
        worker.terminate();
        success ? resolve(result) : reject(error);
      }
    };
    worker.postMessage({ type: "dispatchModel", payload: { preprocessedData } });
  });
}
