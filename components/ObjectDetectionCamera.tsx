import { useRef, useState, useEffect, useCallback } from "react";
import Webcam from "react-webcam";
import { Tensor } from "onnxruntime-web";
import { round } from "lodash";
import ndarray from "ndarray";
import ops from "ndarray-ops";
import { yoloClasses } from "../data/yolo_classes";

const modelResolution = [640, 640];
const modelName = "yolov7-tiny_640x640.onnx";

const WebcamComponent = () => {
  const [inferenceTime, setInferenceTime] = useState<number>(0);
  const [totalTime, setTotalTime] = useState<number>(0);
  const webcamRef = useRef<Webcam>(null);
  const videoCanvasRef = useRef<HTMLCanvasElement>(null);
  const [liveDetection, setLiveDetection] = useState<boolean>(false);
  const [facingMode, setFacingMode] = useState<string>("environment");
  const originalSize = useRef<number[]>([0, 0]);
  const [SSR, setSSR] = useState<Boolean>(true);
  const workerRef = useRef<Worker | null>(null);
  const [outputTensor, setOutputTensor] = useState<Tensor | any>(null);

  useEffect(() => {
    workerRef.current = new Worker(new URL('../utils/modelWorker.ts', import.meta.url), { type: 'module' });
    // Init worker 
    workerRef.current.postMessage({ type: "init" });
    workerRef.current.onmessage = (event) => {
      const { type, success, result, error } = event.data;
      if (type === "dispatchModel") {
        if (success) {
          setOutputTensor(result[0]);
          setInferenceTime(result[1]);
        } else {
          console.error(error);
        }
      }
    };

    // requestAnimationFrame(() => console.log('aaaa'));

    return () => {
      if (workerRef.current) {
        // workerRef.current.terminate();
      }
    }
  }, []);

  const capture = () => {
    const canvas = videoCanvasRef.current!;
    if (!canvas) return;

    const context = canvas.getContext("2d", {
      willReadFrequently: true,
    })!;

    if (facingMode === "user") {
      context.setTransform(-1, 0, 0, 1, canvas.width, 0);
    }

    context.drawImage(
      webcamRef.current!.video!,
      0,
      0,
      canvas.width,
      canvas.height
    );

    if (facingMode === "user") {
      context.setTransform(1, 0, 0, 1, 0, 0);
    }
    return context;
  };

  const resizeCanvasCtx = (
    ctx: CanvasRenderingContext2D,
    targetWidth: number,
    targetHeight: number,
    inPlace = false
  ) => {
    let canvas: HTMLCanvasElement;

    if (inPlace) {
      canvas = ctx.canvas;
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      ctx.scale(
        targetWidth / canvas.clientWidth,
        targetHeight / canvas.clientHeight
      );
    } else {
      canvas = document.createElement("canvas");
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      canvas.getContext("2d")!.drawImage(ctx.canvas, 0, 0, targetWidth, targetHeight);
      ctx = canvas.getContext("2d")!;
    }

    return ctx;
  };

  const preprocess = (ctx: CanvasRenderingContext2D) => {
    const resizedCtx = resizeCanvasCtx(ctx, modelResolution[0], modelResolution[1]);
    const imageData = resizedCtx.getImageData(0, 0, modelResolution[0], modelResolution[1]);
    const { data, width, height } = imageData;
    const dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [1, 3, width, height]);

    ops.assign(dataProcessedTensor.pick(0, 0, null, null), dataTensor.pick(null, null, 0));
    ops.assign(dataProcessedTensor.pick(0, 1, null, null), dataTensor.pick(null, null, 1));
    ops.assign(dataProcessedTensor.pick(0, 2, null, null), dataTensor.pick(null, null, 2));
    ops.divseq(dataProcessedTensor, 255);

    const tensor = new Tensor("float32", new Float32Array(width * height * 3), [1, 3, width, height]);
    (tensor.data as Float32Array).set(dataProcessedTensor.data);
    return tensor;
  };

  const conf2color = (conf: number) => {
    const r = Math.round(255 * (1 - conf));
    const g = Math.round(255 * conf);
    return `rgb(${r},${g},0)`;
  };

  const postprocess = async (tensor: Tensor, ctx: CanvasRenderingContext2D) => {
    const dx = ctx.canvas.width / modelResolution[0];
    const dy = ctx.canvas.height / modelResolution[1];

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    
    if (tensor && tensor.dims.length > 0) {
      for (let i = 0; i < tensor.dims[0]; i++) {
        let [batch_id, x0, y0, x1, y1, cls_id, score] = tensor.data.slice(i * 7, i * 7 + 7);
  
        [x0, x1] = [x0, x1].map((x: any) => x * dx);
        [y0, y1] = [y0, y1].map((x: any) => x * dy);
  
        [batch_id, x0, y0, x1, y1, cls_id] = [batch_id, x0, y0, x1, y1, cls_id].map((x: any) => round(x));
        [score] = [score].map((x: any) => round(x * 100, 1));
  
        const label = yoloClasses[cls_id].toString()[0].toUpperCase() + yoloClasses[cls_id].toString().substring(1) + " " + score.toString() + "%";
        const color = conf2color(score / 100);
  
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
        ctx.font = "20px Arial";
        ctx.fillStyle = color;
        ctx.fillText(label, x0, y0 - 5);
  
        ctx.fillStyle = color.replace(")", ", 0.2)").replace("rgb", "rgba");
        ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      }
    }
  };

  const runModel = async (ctx: CanvasRenderingContext2D) => {
    const data = preprocess(ctx);
    
    if (workerRef.current) {
      workerRef.current.postMessage({ type: "dispatchModel", payload: { preprocessedData: data } })
    }
    if (outputTensor) {
      postprocess(outputTensor, ctx);
    }
  };

  const toggleLiveDetection = useCallback(() => {
    setLiveDetection(prev => !prev);
  }, []);

  // useEffect(() => {
  //   let intervalId: any

  //   if (liveDetection) {
  //     intervalId = setInterval(() => {
  //       const context = capture();
  //       if (context) {
  //         const data = preprocess(context);

  //         if(workerRef.current) {
  //           workerRef.current.postMessage({ type: "dispatchModel", payload: { preprocessedData: data } })
  //         }
  //       }
  //     }, 100)
  //   }

  //   return () => {
  //     if (intervalId) {
  //       clearInterval(intervalId)
  //     }
  //   }
  // }, [liveDetection])

  useEffect(() => {
    if (liveDetection) {
      const startTime = Date.now();
        const ctx = capture();
        if (!ctx) return;
  
        runModel(ctx);
        setTotalTime(Date.now() - startTime);

        const reDraw = async () => {
          await new Promise<void>((resolve) =>
            requestAnimationFrame(() => resolve())
          )
        }
        reDraw()
    }
  }, [liveDetection, runModel]);

  // useEffect(() => {
  //   const context = capture()

  //   if (context) {
  //     postprocess(outputTensor, context)
  //   }
  // }, [outputTensor])

  const processImage = async () => {
    reset();
    const ctx = capture();
    if (!ctx) return;

    const boxCtx = document.createElement("canvas").getContext("2d") as CanvasRenderingContext2D;
    boxCtx.canvas.width = ctx.canvas.width;
    boxCtx.canvas.height = ctx.canvas.height;
    boxCtx.drawImage(ctx.canvas, 0, 0);

    await runModel(boxCtx);
    ctx.drawImage(boxCtx.canvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
  };

  const reset = async () => {
    const context = videoCanvasRef.current!.getContext("2d")!;
    context.clearRect(0, 0, originalSize.current[0], originalSize.current[1]);
    setLiveDetection(false);
  };

  const setWebcamCanvasOverlaySize = () => {
    const element = webcamRef.current!.video!;
    if (!element) return;
    const w = element.offsetWidth;
    const h = element.offsetHeight;
    const cv = videoCanvasRef.current;
    if (!cv) return;
    cv.width = w;
    cv.height = h;
  };

  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        setLiveDetection(false);
      }
      setSSR(document.hidden);
    };
    setSSR(document.hidden);
    document.addEventListener("visibilitychange", handleVisibilityChange);
  }, []);

  if (SSR) {
    return <div>Loading...</div>;
  }

  return (
    <div className="flex flex-row flex-wrap justify-evenly align-center w-full">
      <div id="webcam-container" className="flex items-center justify-center webcam-container">
        <Webcam
          mirrored={facingMode === "user"}
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          imageSmoothing={true}
          videoConstraints={{ facingMode: facingMode }}
          onLoadedMetadata={() => {
            setWebcamCanvasOverlaySize();
            originalSize.current = [
              webcamRef.current!.video!.offsetWidth,
              webcamRef.current!.video!.offsetHeight,
            ] as number[];
          }}
          forceScreenshotSourceSize={true}
        />
        <canvas
          id="cv1"
          ref={videoCanvasRef}
          style={{ position: "absolute", zIndex: 10, backgroundColor: "rgba(0,0,0,0)" }}
        ></canvas>
      </div>
      <div className="flex flex-col justify-center items-center">
        <div className="flex gap-1 flex-row flex-wrap justify-center items-center m-5">
          <div className="flex gap-1 justify-center items-center items-stretch">
            <button
              onClick={async () => {
                const startTime = Date.now();
                await processImage();
                setTotalTime(Date.now() - startTime);
              }}
              className="p-2 border-dashed border-2 rounded-xl hover:translate-y-1 "
            >
              Capture Photo
            </button>
            <button
              onClick={toggleLiveDetection}
              className={`p-2 border-dashed border-2 rounded-xl hover:translate-y-1 ${liveDetection ? "bg-white text-black" : ""}`}
            >
              Live Detection
            </button>
          </div>
          <div className="flex gap-1 justify-center items-center items-stretch">
            <button
              onClick={() => {
                reset();
                setFacingMode(facingMode === "user" ? "environment" : "user");
              }}
              className="p-2 border-dashed border-2 rounded-xl hover:translate-y-1 "
            >
              Switch Camera
            </button>

            <button
              onClick={reset}
              className="p-2 border-dashed border-2 rounded-xl hover:translate-y-1 "
            >
              Reset
            </button>
          </div>
        </div>
        <div>Using {modelName}</div>
        <div className="flex gap-3 flex-row flex-wrap justify-between items-center px-5 w-full">
          <div>
            {"Model Inference Time: " + inferenceTime.toFixed() + "ms"}
            <br />
            {"Total Time: " + totalTime.toFixed() + "ms"}
            <br />
            {"Overhead Time: +" + (totalTime - inferenceTime).toFixed(2) + "ms"}
          </div>
          <div>
            <div>{"Model FPS: " + (1000 / inferenceTime).toFixed(2) + "fps"}</div>
            <div>{"Total FPS: " + (1000 / totalTime).toFixed(2) + "fps"}</div>
            <div>{"Overhead FPS: " + (1000 * (1 / totalTime - 1 / inferenceTime)).toFixed(2) + "fps"}</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WebcamComponent;
