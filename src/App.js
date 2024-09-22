// 1. Install dependencies DONE
// 2. Import dependencies DONE
// 3. Setup webcam and canvas DONE
// 4. Define references to those DONE
// 5. Load posenet DONE
// 6. Detect function DONE
// 7. Drawing utilities from tensorflow DONE
// 8. Draw functions DONE

import React, { useRef, useEffect } from "react";
import "./App.css";
import * as tf from "@tensorflow/tfjs";
import * as posenet from "@tensorflow-models/posenet";
import Webcam from "react-webcam";
import { drawKeypoints, drawSkeleton } from "./utilities";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    const runPoseNet = async () => {
      await tf.setBackend("webgl");
      const net = await posenet.load({
        architecture: 'MobileNetV1',
        outputStride: 16,
        inputResolution: { width: 320, height: 240 },
        multiplier: 0.75
      });

      const detect = async () => {
        if (
          webcamRef.current &&
          webcamRef.current.video.readyState === 4
        ) {
          const video = webcamRef.current.video;
          const videoWidth = video.videoWidth;
          const videoHeight = video.videoHeight;

          webcamRef.current.video.width = videoWidth;
          webcamRef.current.video.height = videoHeight;

          const pose = await net.estimateSinglePose(video, {
            flipHorizontal: false,
          });

          const ctx = canvasRef.current.getContext("2d");
          canvasRef.current.width = videoWidth;
          canvasRef.current.height = videoHeight;

          drawKeypoints(pose.keypoints, 0.6, ctx);
          drawSkeleton(pose.keypoints, 0.6, ctx);
        }
      };

      setInterval(detect, 100); // Throttle frame processing
    };

    runPoseNet();
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <Webcam
          ref={webcamRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
            width: 320,
            height: 240,
          }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
            width: 320,
            height: 240,
          }}
        />
      </header>
    </div>
  );
}

export default App;
