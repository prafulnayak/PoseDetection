import React, { useRef, useEffect } from "react";
import "./App.css";
import * as tf from "@tensorflow/tfjs";
import * as posenet from "@tensorflow-models/posenet";
import Webcam from "react-webcam";
import { drawKeypoints, drawSkeleton, isMobile } from "./utilities";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const handRaiseCountRef = useRef(0);
  const prevHandRaisedRef = useRef(false);

  const videoConstraints = isMobile()
    ? { facingMode: { exact: "environment" } } // Back camera for mobile devices
    : { facingMode: "user" }; // Front camera for laptops

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

          // Check if hand is raised above shoulder with confidence >= 0.6
          const leftWrist = pose.keypoints.find(point => point.part === 'leftWrist');
          const rightWrist = pose.keypoints.find(point => point.part === 'rightWrist');
          const leftShoulder = pose.keypoints.find(point => point.part === 'leftShoulder');
          const rightShoulder = pose.keypoints.find(point => point.part === 'rightShoulder');

          const handRaised = (
            (leftWrist.score >= 0.6 && leftShoulder.score >= 0.6 && leftWrist.position.y < leftShoulder.position.y) ||
            (rightWrist.score >= 0.6 && rightShoulder.score >= 0.6 && rightWrist.position.y < rightShoulder.position.y)
          );

          if (handRaised && !prevHandRaisedRef.current) {
            handRaiseCountRef.current += 1;
            console.log("Hand raised ${handRaiseCountRef.current} times");
          }

          prevHandRaisedRef.current = handRaised;
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
            zIndex: 9,
            width: 320,
            height: 240,
          }}
          videoConstraints={videoConstraints}
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
            zIndex: 9,
            width: 320,
            height: 240,
          }}
        />
        <div style={{ position: "absolute", top: 10, left: 10, zIndex: 10, color: "white" }}>
          Hand Raise Count: {handRaiseCountRef.current}
        </div>
      </header>
    </div>
  );
}

export default App;
