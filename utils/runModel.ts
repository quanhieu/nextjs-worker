import { Tensor } from "onnxruntime-web";

export function createModelCpu(worker: Worker): void {
  worker.postMessage = () => {
    console.log('Init worker')
  };
}

// export async function dispatchModel(worker: any, preprocessedData: Tensor): Promise<[Tensor, number]> {
//   return new Promise((resolve, reject) => {
//     worker.onmessage = (event: any) => {
//       const { type, success, result, error } = event.data;
//       if (type === "dispatchModel") {
//         // worker.terminate();
//         success ? resolve(result) : reject(error);
//       }
//     };
//     worker.postMessage({ type: "dispatchModel", payload: { preprocessedData } });
//   });
// }

