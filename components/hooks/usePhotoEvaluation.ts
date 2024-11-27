import Toast, { ToastOptions } from "react-native-root-toast";
import {
  EvaluatePhotoClient,
  EvaluationResults,
} from "../network/EvaluatePhotoClient";

export function usePhotoEvaluation() {
  return evaluatePhotoAndDisplayResults;
}

async function evaluatePhotoAndDisplayResults(photoBase64?: string) {
  const evaluationResults = await new EvaluatePhotoClient().evaluatePhoto(
    photoBase64
  );

  try {
    if (evaluationResults) showResultsToast(evaluationResults);
    else throw new Error("Failed to evaluate photo.");
  } catch (error) {
    console.error("Failed to take photo:", error);
    showEvaluatePhotoErrorToast();
  }

  return evaluationResults;
}

const getToastOptions = (
  positionOffset: number,
  value?: number
): ToastOptions => ({
  duration: 3000,
  position: positionOffset,
  shadow: true,
  animation: true,
  hideOnPress: false,
  backgroundColor: colorFromValue(value ?? 0),
});

function getPosition(value: number) {
  return Toast.positions.BOTTOM - 50 * value;
}

function getPredictionToastOptions(
  positionOffset: number,
  color: string
): ToastOptions | undefined {
  return {
    duration: 3000,
    position: positionOffset,
    shadow: true,
    animation: true,
    hideOnPress: false,
    backgroundColor: color,
    textColor: "black",
  };
}

function showResultsToast(results: EvaluationResults) {
  const prediction = getPrediction(results);

  Toast.show(
    "Prediction: " + prediction,
    getPredictionToastOptions(getPosition(3), "lightblue")
  );

  Toast.show(
    `😻 ${toPercent(results.captain)}%`,
    getToastOptions(getPosition(2), results.captain)
  );

  Toast.show(
    `😼 ${toPercent(results.bathroom_cat)}%`,
    getToastOptions(getPosition(1), results.bathroom_cat)
  );

  Toast.show(
    `🪑 ${toPercent(results.control)}%`,
    getToastOptions(getPosition(0), results.control)
  );
}

function colorFromValue(value: number): string | undefined {
  const red = Math.floor(255 * (1 - value));
  const green = Math.floor(255 * value);

  return `rgb(${red}, ${green}, 0)`;
}

function showEvaluatePhotoErrorToast() {
  Toast.show("❌ Failed to Evaluate Photo.", {
    duration: 1500,
    position: Toast.positions.BOTTOM,
    shadow: true,
    animation: true,
    hideOnPress: true,
    backgroundColor: "red",
  });
}

const getPrediction = (results: EvaluationResults) => {
  if (!results) return "No prediction available.";

  const max = Math.max(
    results.captain ?? -1,
    results.bathroom_cat ?? -1,
    results.control ?? -1
  );

  if (max === results.captain) return "😻";
  if (max === results.bathroom_cat) return "😼";
  if (max === results.control) return "🪑";
  return "No prediction available.";
};

function toPercent(captain: number | undefined) {
  return captain ? Math.round(captain * 100) : 0;
}
